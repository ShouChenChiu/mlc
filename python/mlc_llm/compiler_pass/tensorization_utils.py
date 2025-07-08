import tvm
from tvm import IRModule, relax, tir
from tvm.contrib import clang, utils
from tvm.script import tir as T


import tvm
from tvm import tir
from tvm.ir import IRModule
from tvm.tir import stmt_functor
import tvm.tir.transform as TTransform

def inject_dyn_Symbol(func: tir.PrimFunc, mod: IRModule, ctx):
    a_handle = func.params[2]          
    A_buf = func.buffer_map[a_handle]  
    M = A_buf.shape[1]
    def post(node):
        if isinstance(node, tir.Call) and getattr(node.op, "name", "") == "tir.call_extern":
            if "mm" in str(list(node.args)[0]): 
                try:
                    from tvm.ir import structural_equal
                    if not any(structural_equal(arg, M) for arg in node.args):
                        args = list(node.args) + [M]
                    return tir.Call(node.dtype, node.op, args, node.span)
                except:
                    pass
        return node
    new_body = stmt_functor.ir_transform(
        func.body,
        None,
        post
    )
    return tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
        span=func.span
    )

@TTransform.prim_func_pass(opt_level=1)
def inject_dyn_pass(func, mod, ctx):
    return inject_dyn_Symbol(func, mod, ctx)

kernel_dict = {}
kernel_dict["deq_packing_32x32"] = """
#include <riscv_vector.h>
extern "C" void deq_packing_32x32( uint32_t* weight,  int w_offset,
                                   float* scale, int s_offset,
                                   float* output, int out_offset,
                                   int K) {
    size_t vl = __riscv_vsetvl_e32m4(32);
    vuint32m4_t u0, p0;
    vfloat32m4_t f0, s0;
    // using byte as stride 
    ptrdiff_t w_stride = K / 8 * 4; 
    ptrdiff_t s_stride = K / 32 * 4;        
    s0 = __riscv_vlse32_v_f32m4(scale + s_offset, s_stride, vl);
    for(int i = 0 ; i < 4 ; i ++){
        p0 = __riscv_vlse32_v_u32m4(weight + w_offset + i, w_stride, vl);
        for(int j = 0 ; j < 8 ; j++){
            u0 = __riscv_vsrl_vx_u32m4(p0, j * 4, vl);
            u0 = __riscv_vand_vx_u32m4(u0, 15, vl);
            f0 = __riscv_vfcvt_f_xu_v_f32m4(u0, vl);
            f0 = __riscv_vfsub_vf_f32m4(f0, 7.0, vl);
            f0 = __riscv_vfmul_vv_f32m4(f0, s0, vl);
            __riscv_vse32_v_f32m4(output + out_offset + j * 32 + i * 256, f0, vl);
        }
    }
}
"""


kernel_dict["packed_mm_4x32"] = """
#include <riscv_vector.h>
extern "C" void packed_mm_4x32(float* out, int out_offset,
                                  float* lhs, int lhs_offset,
                                  float* rhs, int rhs_offset,
                                  int64_ N, int K, int64_t M) {
    // compute residual M dim
    int res_M = M - lhs_offset / K; 
    vfloat32m4_t acc0, acc1, acc2, acc3, vb;
    size_t vl = __riscv_vsetvl_e32m4(32);
    // perform 4x32 outer-product
    if( res_M  >= 4){
        acc0 = __riscv_vle32_v_f32m4(out + out_offset + 0 * N, vl);
        acc1 = __riscv_vle32_v_f32m4(out + out_offset + 1 * N, vl);
        acc2 = __riscv_vle32_v_f32m4(out + out_offset + 2 * N, vl);
        acc3 = __riscv_vle32_v_f32m4(out + out_offset + 3 * N, vl);

        for(int k = 0; k < 128; k++){
            vb = __riscv_vle32_v_f32m4(rhs + rhs_offset + k * 32, vl);
            acc0 = __riscv_vfmacc_vf_f32m4(acc0, *(lhs + lhs_offset + k + 0 * K), vb, vl);
            acc1 = __riscv_vfmacc_vf_f32m4(acc1, *(lhs + lhs_offset + k + 1 * K), vb, vl);
            acc2 = __riscv_vfmacc_vf_f32m4(acc2, *(lhs + lhs_offset + k + 2 * K), vb, vl);
        }
        __riscv_vse32_v_f32m4(out + out_offset + 0 * N, acc0, vl);
        __riscv_vse32_v_f32m4(out + out_offset + 1 * N, acc1, vl);
        __riscv_vse32_v_f32m4(out + out_offset + 2 * N, acc2, vl);
        __riscv_vse32_v_f32m4(out + out_offset + 3 * N, acc3, vl);
    }
    else{ // fallback to length = 1 
        for(int i = 0 ; i < res_M ; i++){
            acc0 = __riscv_vle32_v_f32m4(out + out_offset + i * N, vl);
            for(int k = 0; k < 128; k++){
                vb = __riscv_vle32_v_f32m4(rhs + rhs_offset + k * 32, vl);
                acc0 = __riscv_vfmacc_vf_f32m4(acc0, *(lhs + lhs_offset + k + i * K), vb, vl);
            }
            __riscv_vse32_v_f32m4(out + out_offset + i * N, acc0, vl);
        }
    }
}
"""



kernel_dict["sgemm_8x16"] = """
#include <riscv_vector.h>
extern "C" void sgemm_8x16(float *out, const float *a, const float *b, const int K, const float *scale) {
    vfloat32m2_t  acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    size_t vl = __riscv_vsetvl_e32m2(16);

    // Init OUT zeros 
    acc0 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc1 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc2 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc3 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc4 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc5 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc6 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc7 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    // MATMUL outer product along K 
    // Take Query[0:7,0:K] cross Key[0:K, 0:vl]
    for(int k = 0 ; k < K; ++k){
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + 16 * k, vl);
        acc0 = __riscv_vfmacc_vf_f32m2(acc0, *(a + k + K * 0), vb, vl);
        acc1 = __riscv_vfmacc_vf_f32m2(acc1, *(a + k + K * 1), vb, vl);
        acc2 = __riscv_vfmacc_vf_f32m2(acc2, *(a + k + K * 2), vb, vl);
        acc3 = __riscv_vfmacc_vf_f32m2(acc3, *(a + k + K * 3), vb, vl);
        acc4 = __riscv_vfmacc_vf_f32m2(acc4, *(a + k + K * 4), vb, vl);
        acc5 = __riscv_vfmacc_vf_f32m2(acc5, *(a + k + K * 5), vb, vl);
        acc6 = __riscv_vfmacc_vf_f32m2(acc6, *(a + k + K * 6), vb, vl);
        acc7 = __riscv_vfmacc_vf_f32m2(acc7, *(a + k + K * 7), vb, vl);
    }
    
    // Mul with Scale
    acc0 = __riscv_vfmul_vf_f32m2(acc0, scale[0], vl);
    acc1 = __riscv_vfmul_vf_f32m2(acc1, scale[0], vl);
    acc2 = __riscv_vfmul_vf_f32m2(acc2, scale[0], vl);
    acc3 = __riscv_vfmul_vf_f32m2(acc3, scale[0], vl);
    acc4 = __riscv_vfmul_vf_f32m2(acc4, scale[0], vl);
    acc5 = __riscv_vfmul_vf_f32m2(acc5, scale[0], vl);
    acc6 = __riscv_vfmul_vf_f32m2(acc6, scale[0], vl); 
    acc7 = __riscv_vfmul_vf_f32m2(acc7, scale[0], vl); 

    // Store out
    const int out_strides = 16;
    __riscv_vse32_v_f32m2(out + out_strides * 0, acc0, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 1, acc1, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 2, acc2, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 3, acc3, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 4, acc4, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 5, acc5, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 6, acc6, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 7, acc7, vl);
}
"""

kernel_dict["ogemm_8x64"] = """
#include <riscv_vector.h>
extern "C" void ogemm_8x64( float *out, const int out_offset, const int out_strides,
                float *a, const int a_offset, const int a_strides,
                float *b, const int b_offset, const int b_strides){

    size_t vl = __riscv_vsetvl_e32m2(16);
    vfloat32m2_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    for(int t = 0 ; t < 4 ; t++){
        acc0 = __riscv_vle32_v_f32m2(out + out_offset + out_strides * 0 + 16 * t, vl);
        acc1 = __riscv_vle32_v_f32m2(out + out_offset + out_strides * 1 + 16 * t, vl);
        acc2 = __riscv_vle32_v_f32m2(out + out_offset + out_strides * 2 + 16 * t, vl);
        acc3 = __riscv_vle32_v_f32m2(out + out_offset + out_strides * 3 + 16 * t, vl);
        acc4 = __riscv_vle32_v_f32m2(out + out_offset + out_strides * 4 + 16 * t, vl);
        acc5 = __riscv_vle32_v_f32m2(out + out_offset + out_strides * 5 + 16 * t, vl);
        acc6 = __riscv_vle32_v_f32m2(out + out_offset + out_strides * 6 + 16 * t, vl);
        acc7 = __riscv_vle32_v_f32m2(out + out_offset + out_strides * 7 + 16 * t, vl);
        for (int i = 0 ; i < 16 ; i++ ){
            vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + b_offset + b_strides * i + 16 * t, vl);
            acc0 = __riscv_vfmacc_vf_f32m2(acc0, *(a + i + a_strides * 0), vb, vl);
            acc1 = __riscv_vfmacc_vf_f32m2(acc1, *(a + i + a_strides * 1), vb, vl);
            acc2 = __riscv_vfmacc_vf_f32m2(acc2, *(a + i + a_strides * 2), vb, vl);
            acc3 = __riscv_vfmacc_vf_f32m2(acc3, *(a + i + a_strides * 3), vb, vl);
            acc4 = __riscv_vfmacc_vf_f32m2(acc4, *(a + i + a_strides * 4), vb, vl);
            acc5 = __riscv_vfmacc_vf_f32m2(acc5, *(a + i + a_strides * 5), vb, vl);
            acc6 = __riscv_vfmacc_vf_f32m2(acc6, *(a + i + a_strides * 6), vb, vl);
            acc7 = __riscv_vfmacc_vf_f32m2(acc7, *(a + i + a_strides * 7), vb, vl);
        }
        __riscv_vse32_v_f32m2(out + out_offset + out_strides * 0 + 16 * t, acc0, vl);
        __riscv_vse32_v_f32m2(out + out_offset + out_strides * 1 + 16 * t, acc1, vl);
        __riscv_vse32_v_f32m2(out + out_offset + out_strides * 2 + 16 * t, acc2, vl);
        __riscv_vse32_v_f32m2(out + out_offset + out_strides * 3 + 16 * t, acc3, vl);
        __riscv_vse32_v_f32m2(out + out_offset + out_strides * 4 + 16 * t, acc4, vl);
        __riscv_vse32_v_f32m2(out + out_offset + out_strides * 5 + 16 * t, acc5, vl);
        __riscv_vse32_v_f32m2(out + out_offset + out_strides * 6 + 16 * t, acc6, vl);
        __riscv_vse32_v_f32m2(out + out_offset + out_strides * 7 + 16 * t, acc7, vl);
    }   

}"""

kernel_dict["mmt_1x2x64"] =  """
    #include <riscv_vector.h>
    extern "C" void mmt_1x2x64(float *out, const int out_offset, const int out_strides,
        float *a, const int a_offset, const int a_strides,
        float *b, const int b_offset, const int b_strides){

        size_t vl = __riscv_vsetvl_e32m4(32);
        
        vfloat32m4_t va0 = __riscv_vle32_v_f32m4(a + a_offset, vl);
        vfloat32m4_t va1 = __riscv_vle32_v_f32m4(a + a_offset + 32, vl);

        float *b_ptr = b + b_offset;
        float *out_ptr = out + out_offset;
        vfloat32m4_t vb0;
        for(int i = 0 ; i < 2 ; ++i){
            vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.f, vl);
            vb0 = __riscv_vle32_v_f32m4(b_ptr, vl);
            acc0 = __riscv_vfmacc_vv_f32m4(acc0, va0, vb0, vl);
            vb0 = __riscv_vle32_v_f32m4(b_ptr + 32, vl);
            acc0 = __riscv_vfmacc_vv_f32m4(acc0, va1, vb0, vl);
            b_ptr += b_strides;
            *(out_ptr + i) +=  __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc0, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl));
        }
    }
"""

kernel_dict["mmt_1x4x128"] =  """
    #include <riscv_vector.h>
    extern "C" void mmt_1x4x128( float *out, const int out_offset, const int out_strides,
                    float *a, const int a_offset, const int a_strides,
                    float *b, const int b_offset, const int b_strides,
                    int K){

        size_t vl = __riscv_vsetvl_e32m4(32);
        vfloat32m4_t  va0, va1, va2, va3, acc0;
        vfloat32m1_t zeros = __riscv_vfmv_s_f_f32m1(0.0f, 8);

        va0 = __riscv_vle32_v_f32m4(a + a_offset + vl * 0, vl);
        va1 = __riscv_vle32_v_f32m4(a + a_offset + vl * 1, vl);
        va2 = __riscv_vle32_v_f32m4(a + a_offset + vl * 2, vl);
        va3 = __riscv_vle32_v_f32m4(a + a_offset + vl * 3, vl);
        
        int N = 4;
        for(int n = 0; n < N ; n++){
            acc0 = __riscv_vfmv_v_f_f32m4(0.f, vl);
            vfloat32m4_t vb0 = __riscv_vle32_v_f32m4(b + b_offset + b_strides * n + vl * 0, vl);
            acc0 = __riscv_vfmacc_vv_f32m4(acc0, va0, vb0, vl);

            vb0 = __riscv_vle32_v_f32m4(b + b_offset + b_strides * n + vl * 1, vl);
            acc0 = __riscv_vfmacc_vv_f32m4(acc0, va1, vb0, vl);

            vb0 = __riscv_vle32_v_f32m4(b + b_offset + b_strides * n + vl * 2, vl);
            acc0 = __riscv_vfmacc_vv_f32m4(acc0, va2, vb0, vl);

            vb0 = __riscv_vle32_v_f32m4(b + b_offset + b_strides * n + vl * 3, vl);
            acc0 = __riscv_vfmacc_vv_f32m4(acc0, va3, vb0, vl);

            *(out + out_offset + n) += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc0, zeros, vl));
        }
    }
"""

def get_ll_code(kername):
    temp = utils.tempdir()
    ll_path = temp.relpath("test.ll")
    return clang.create_llvm(inputs=kernel_dict[kername], 
                              output=ll_path,
                              options=["--target=riscv64-linux-unknown-gnu", 
                                        "--sysroot=/home/scchiu/toolchain/sysroot", 
                                        "--gcc-toolchain=/home/scchiu/toolchain", 
                                        "-march=rv64gcv", "-c"]
                            )
    
def get_kername(m_size, n_size, blk_name, k_size = 0):
    if k_size == 0:
        return blk_name + str(m_size) + "x" + str(n_size)
    else:
        return blk_name + str(m_size) + "x" + str(n_size) + "x" + str(k_size)

def sgemm_register(m_size, n_size, k_size):
    ker_name = get_kername(m_size, n_size, "sgemm_")
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle, s: T.handle) -> None:
        A = T.match_buffer(a, (m_size, k_size), align=64, offset_factor=1)
        B = T.match_buffer(b, (k_size, n_size), align=64, offset_factor=1)
        C = T.match_buffer(c, (m_size, n_size), align=64, offset_factor=1)
        S = T.match_buffer(s, (1,))
        with T.block("root"):
            T.reads(A[0 : m_size, 0 : k_size], B[0 : k_size, 0 : n_size], S[0])
            T.writes(C[0 : m_size, 0 : n_size])
            with T.init():
                for vi, vj in T.grid(m_size, n_size):
                    with T.block("gemm_init"):
                        i, j = T.axis.remap("SS", [vi, vj])
                        T.reads()
                        T.writes(C[i, j])
                        C[i, j] = T.float32(0.0)
            for vi, vj, vk in T.grid(m_size, n_size, k_size):
                with T.block("gemm"):
                    i, j, k = T.axis.remap("SSR", [vi, vj, vk])
                    T.reads(C[i, j], A[i, k], B[k, j], S[0])
                    T.writes(C[i, j])
                    C[i, j] = C[i, j] + A[i, k] * B[k, j] * S[0]

    @T.prim_func
    def impl(a: T.handle, b: T.handle, c: T.handle, s: T.handle) -> None:
        A = T.match_buffer(a, (m_size, k_size), align=64, offset_factor=1)
        B = T.match_buffer(b, (k_size, n_size), align=64, offset_factor=1)
        C = T.match_buffer(c, (m_size, n_size), align=64, offset_factor=1)
        S = T.match_buffer(s, (1,))
        with T.block("root"):
            T.reads(A[0 : m_size, 0 : k_size], B[0 : k_size, 0 : n_size], S[0])
            T.writes(C[0 : m_size, 0 : n_size])
            T.evaluate(
                T.call_extern(
                    "void",                   
                    ker_name,           
                    C.data, A.data, B.data, k_size, S.data
                )
            )
    tir.TensorIntrin.register(ker_name, desc, impl)
    return ker_name

def ogemm_register(m_size, n_size, k_size):
    ker_name = get_kername(m_size, n_size, "ogemm_")
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (m_size, k_size), align=64, offset_factor=1)
        B = T.match_buffer(b, (k_size, n_size), align=64, offset_factor=1)
        C = T.match_buffer(c, (m_size, n_size), align=64, offset_factor=1)
        with T.block("root"):
            T.reads(C[0 : m_size, 0 : n_size], A[0 : m_size, 0 : k_size], B[0 : k_size, 0 : n_size])
            T.writes(C[0 : m_size, 0 : n_size])
            for vi, vj, vk in T.grid(m_size, n_size, k_size):
                with T.block("gemm"):
                    i, j, k = T.axis.remap("SSR", [vi, vj, vk])
                    T.reads(C[i, j], A[i, k], B[k, j])
                    T.writes(C[i, j])
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

    @T.prim_func
    def intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (m_size, k_size), align=64, offset_factor=1)
        B = T.match_buffer(b, (k_size, n_size), align=64, offset_factor=1)
        C = T.match_buffer(c, (m_size, n_size), align=64, offset_factor=1)
        with T.block("root"):
            T.reads(C[0 : m_size, 0 : n_size], A[0 : m_size, 0 : k_size], B[0 : k_size, 0 : n_size])
            T.writes(C[0 : m_size, 0 : n_size])
            T.evaluate(
                T.call_extern(
                    "void",                   
                    ker_name,           
                    C.data, C.elem_offset, 64,
                    A.data, A.elem_offset, 16,
                    B.data, B.elem_offset, 64,
                )
            )
    tir.TensorIntrin.register(ker_name, desc, intrin)
    return ker_name

def seq_mmt_register(m_size, n_size, k_size, K, N): 
    ker_name = get_kername(m_size, n_size, "mmt_", k_size)
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (1, m_size, k_size), align=64, offset_factor=1)
        B = T.match_buffer(b, (n_size, k_size), align=64, offset_factor=1)
        C = T.match_buffer(c, (1, m_size, n_size), align=64, offset_factor=1)
        with T.block("root"):
            T.reads(A[0, 0 : m_size, 0 : k_size], B[0 : n_size, 0 : k_size])
            T.writes(C[0, 0 : m_size, 0 : n_size])
            for vj, vk in T.grid(n_size, k_size):
                with T.block("gemm"):
                    j, k = T.axis.remap("SR", [vj, vk])
                    with T.init():
                        C[0, 0, j] == T.float32(0.0)
                    T.reads(C[0, 0, j], A[0, 0, k], B[j, k])
                    T.writes(C[0, 0, j])
                    C[0, 0, j] = C[0, 0, j] + A[0, 0, k] * B[j, k]

    @T.prim_func
    def intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (1, m_size, k_size), align=64, offset_factor=1)
        B = T.match_buffer(b, (n_size, k_size), align=64, offset_factor=1)
        C = T.match_buffer(c, (1, m_size, n_size), align=64, offset_factor=1)
        with T.block("root"):
            T.reads(A[0, 0 : m_size, 0 : k_size], B[0 : n_size, 0 : k_size])
            T.writes(C[0, 0 : m_size, 0 : n_size])
            T.evaluate(
                T.call_extern(
                    "void",                   
                    ker_name,           
                    C.data, C.elem_offset, N,
                    A.data, A.elem_offset, K,
                    B.data, B.elem_offset, K,
                )
            )
    tir.TensorIntrin.register(ker_name, desc, intrin, override=True)
    return ker_name

def batch_mmt_register(m_size, n_size, k_size, K, N): 
    ker_name = get_kername(m_size, n_size, "mmt_", k_size)
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (m_size, 1, k_size), align=64, offset_factor=1)
        B = T.match_buffer(b, (n_size, k_size), align=64, offset_factor=1)
        C = T.match_buffer(c, (m_size, 1, n_size), align=64, offset_factor=1)
        with T.block("root"):
            T.reads(A[0 : m_size, 0, 0 : k_size], B[0 : n_size, 0 : k_size])
            T.writes(C[0 : m_size, 0, 0 : n_size])
            for vj, vk in T.grid(n_size, k_size):
                with T.block("gemm"):
                    j, k = T.axis.remap("SR", [vj, vk])
                    with T.init():
                        C[0, 0, j] == T.float32(0.0)
                    T.reads(C[0, 0, j], A[0, 0, k], B[j, k])
                    T.writes(C[0, 0, j])
                    C[0, 0, j] = C[0, 0, j] + A[0, 0, k] * B[j, k]

    @T.prim_func
    def intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (m_size, 1, k_size), align=64, offset_factor=1)
        B = T.match_buffer(b, (n_size, k_size), align=64, offset_factor=1)
        C = T.match_buffer(c, (m_size, 1, n_size), align=64, offset_factor=1)
        with T.block("root"):
            T.reads(A[0 : m_size, 0, 0 : k_size], B[0 : n_size, 0 : k_size])
            T.writes(C[0 : m_size, 0, 0 : n_size])
            T.evaluate(
                T.call_extern(
                    "void",                   
                    ker_name,           
                    C.data, C.elem_offset, N,
                    A.data, A.elem_offset, K,
                    B.data, B.elem_offset, K,
                )
            )
    tir.TensorIntrin.register(ker_name, desc, intrin, override=True)
    return ker_name

def deq_register(K, tile):
    ker_name = "deq_packing_32x32"
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (tile, 4), dtype="uint32",offset_factor=1)
        B = T.match_buffer(b, (tile, 1), offset_factor=1)
        C = T.match_buffer(c, (1, 32, tile), offset_factor=1)
        with T.block("root"):
            T.reads(A[0 : tile, 0 : 4], B[0: tile, 0] )
            T.writes(C[0, 0 : 32, 0 : tile])
            for vi, vj in T.grid(T.int64(tile), T.int64(32)):
                with T.block("inline_deq"):
                    i, j = T.axis.remap("SS", [vi, vj])
                    T.reads(A[i, j // 8], B[i, j // 32])
                    T.writes(C[0, j, i])
                    C[0, j, i] = (T.Cast("float32", T.bitwise_and(T.shift_right(A[i, j // T.int64(8)], T.Cast("uint32", j % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float32(7.0)) * B[i, j // T.int64(32)]

    @T.prim_func
    def intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (tile, 4), dtype="uint32",offset_factor=1)
        B = T.match_buffer(b, (tile, 1), offset_factor=1)
        C = T.match_buffer(c, (1, 32, tile), offset_factor=1)
        with T.block("root"):
            T.reads(A[0 : tile, 0 : 4],B[0: tile, 0] )
            T.writes(C[0, 0 : 32, 0 : tile])
            T.evaluate(
                T.call_extern(
                    "void",                   
                    ker_name,           
                    A.data, A.elem_offset,
                    B.data, B.elem_offset,
                    C.data, C.elem_offset,
                    K
                )
            )
    tir.TensorIntrin.register(ker_name, desc, intrin, override=True)
    return ker_name

def seq_packed_mm_register(m_size, n_size, k_size, K, N): 
    ker_name = "packed_mm_4x32"
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (1, m_size, k_size), offset_factor=1)
        B = T.match_buffer(b, (1, k_size, n_size), offset_factor=1)
        C = T.match_buffer(c, (1, m_size, n_size), offset_factor=1)
        with T.block("root"):
            T.reads(C[0, 0 : m_size, 0 : n_size] ,A[0, 0 : m_size, 0 : k_size], B[0, 0 : k_size, 0 : n_size])
            T.writes(C[0, 0 : m_size, 0 : n_size])
            for vi, vj, vk in T.grid(m_size, n_size, k_size):
                with T.block("gemm"):
                    i, j, k = T.axis.remap("SSR", [vi, vj, vk])
                    T.reads(C[0, i, j], A[0, i, k], B[0, k, j])
                    T.writes(C[0, i, j])
                    C[0, i, j] = C[0, i, j] + A[0, i, k] * B[0, k, j]

    @T.prim_func
    def intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (1, m_size, k_size), offset_factor=1)
        B = T.match_buffer(b, (1, k_size, n_size), offset_factor=1)
        C = T.match_buffer(c, (1, m_size, n_size), offset_factor=1)
        with T.block("root"):
            T.reads(C[0, 0 : m_size, 0 : n_size] ,A[0, 0 : m_size, 0 : k_size], B[0, 0 : k_size, 0 : n_size])
            T.writes(C[0, 0 : m_size, 0 : n_size])
            T.evaluate(
                T.call_extern(
                    "void",                   
                    ker_name,           
                    C.data, C.elem_offset,
                    A.data, A.elem_offset,
                    B.data, B.elem_offset,
                    N, K
                )
            )
    tir.TensorIntrin.register(ker_name, desc, intrin, override=True)
    return ker_name

def batch_packed_mm_register(m_size, n_size, k_size, K, N): 
    ker_name = "packed_mm_4x32"
    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (m_size, 1, k_size), offset_factor=1)
        B = T.match_buffer(b, (1, k_size, n_size), offset_factor=1)
        C = T.match_buffer(c, (m_size, 1, n_size), offset_factor=1)
        with T.block("root"):
            T.reads(C[0 : m_size, 0, 0 : n_size] ,A[0 : m_size, 0, 0 : k_size], B[0, 0 : k_size, 0 : n_size])
            T.writes(C[0 : m_size, 0, 0 : n_size])
            for vi, vj, vk in T.grid(m_size, n_size, k_size):
                with T.block("gemm"):
                    i, j, k = T.axis.remap("SSR", [vi, vj, vk])
                    T.reads(C[i, 0, j], A[i, 0, k], B[0, k, j])
                    T.writes(C[i, 0, j])
                    C[i, 0, j] = C[i, 0, j] + A[i, 0, k] * B[0, k, j]

    @T.prim_func
    def intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (m_size, 1, k_size), offset_factor=1)
        B = T.match_buffer(b, (1, k_size, n_size), offset_factor=1)
        C = T.match_buffer(c, (m_size, 1, n_size), offset_factor=1)
        with T.block("root"):
            T.reads(C[0 : m_size, 0, 0 : n_size] ,A[0 : m_size, 0, 0 : k_size], B[0, 0 : k_size, 0 : n_size])
            T.writes(C[0 : m_size, 0, 0 : n_size])
            T.evaluate(
                T.call_extern(
                    "void",                   
                    ker_name,           
                    C.data, C.elem_offset,
                    A.data, A.elem_offset,
                    B.data, B.elem_offset,
                    N, K
                )
            )
    tir.TensorIntrin.register(ker_name, desc, intrin, override=True)
    return ker_name