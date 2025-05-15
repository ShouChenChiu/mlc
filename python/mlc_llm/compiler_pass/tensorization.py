"""The pass that attaches logit processor functions to the IRModule."""

import tvm
from tvm import IRModule
from tvm.script import tir as T
from tvm import IRModule, relax, tir

from mlc_llm.support import logging

logger = logging.getLogger(__name__)

@tvm.transform.module_pass(opt_level=0, name="TensorizePrefill")
class TensorizePrefill:  # pylint: disable=too-few-public-methods
    """Tensorize GEMM in Prefill_paged_kv_cpu to IRModule."""

    def __init__(self, target: tvm.target.Target):
        self.target = target

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        if self.target.kind.name == "llvm":
            if "-mtriple=riscv64-unknown-linux-gnu" in str(self.target) and "+v" in str(self.target):
                for gv, func in mod.functions_items():
                    if gv.name_hint == "batch_prefill_paged_kv_cpu" and isinstance(func, tir.PrimFunc) :
                        sch_mod = tvm.IRModule({"main": func})
                        sch = tir.Schedule(sch_mod)
                        sch = apply_to_Sgemm(sch, gv.name_hint)
                        #sch = apply_to_Ogemm(sch, gv.name_hint)
                        mod[gv] = sch.mod["main"]
        return mod
        
def get_kername(m_size, n_size, blk_name):
    return blk_name + str(m_size) + "x" + str(n_size)

def sgemm_register(m_size, n_size, k_size):
    ker_name = get_kername(m_size, n_size, "sgemm_")
    from tvm.script import tir as T
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
    def intrin(a: T.handle, b: T.handle, c: T.handle, s: T.handle) -> None:
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
    tir.TensorIntrin.register(ker_name, desc, intrin)
    return ker_name

def ogemm_register(m_size, n_size, k_size):
    ker_name = get_kername(m_size, n_size, "ogemm_")
    from tvm.script import tir as T
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
                    C.data, A.data, B.data, k_size
                )
            )
    tir.TensorIntrin.register(ker_name, desc, intrin)
    return ker_name

def apply_to_Sgemm(sch: tir.Schedule, prefill):
    s_blk = sch.get_block("S_gemm")
    block = sch.get(s_blk)
    Q_buffer = block.reads[0].buffer
    KV_buffer = block.reads[1].buffer
    m_size, k_size = Q_buffer.shape  # Q = [m, k]
    n_size, nk_size = KV_buffer.shape # KV = [n, k]
    # transpose KV buffer to row major 
    if k_size == nk_size and k_size != n_size:
        sch.transform_layout(s_blk, buffer=("read",1),
                index_map = lambda m,n: (n, m))
    elif k_size == n_size and k_size != nk_size:
        n_size = nk_size

    # regist rvv tensor instrin
    ker_name = sgemm_register(m_size, n_size, k_size)

    # tensorize loop_x
    loop_x, loop_y, loop_z = sch.get_loops(s_blk)[-3:]
    sch.tensorize(loop_x, ker_name)
    logger.info(
        "[Tensorize] Function `%s` : %s",
        prefill,
        ker_name
    )
    return sch

def apply_to_Ogemm(sch: tir.Schedule, prefill):
    s_blk = sch.get_block("O_gemm")
    block = sch.get(s_blk)
    Q_buffer = block.reads[1].buffer
    KV_buffer = block.reads[2].buffer
    m_size, k_size = Q_buffer.shape  # Q = [m, k]
    n_size, nk_size = KV_buffer.shape # KV = [n, k]
    # transpose KV buffer to row major 
    if k_size == nk_size and k_size != n_size:
        sch.transform_layout(s_blk, buffer=("read",2),
                index_map = lambda m,n: (n, m))
    elif k_size == n_size and k_size != nk_size:
        n_size = nk_size
    # regist rvv tensor instrin with 2x64
    ker_name = ogemm_register(2, 64, k_size)
    # tensorize loop_x
    loop_x, loop_y, loop_z = sch.get_loops(s_blk)[-3:]
    x_o, x_i = sch.split(loop_x, [None, 2])
    sch.tensorize(x_i, ker_name)
    logger.info(
        "[Tensorize] Function `%s` : %s",
        prefill,
        ker_name
    )
    return sch