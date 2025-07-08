"""The pass that attaches logit processor functions to the IRModule."""
import tvm
from tvm import IRModule
from tvm.script import tir as T
from tvm import IRModule, relax, tir

from mlc_llm.support import logging
from .tensorization_utils import get_ll_code, get_kername, sgemm_register, ogemm_register, seq_mmt_register,batch_mmt_register, inject_dyn_pass, seq_packed_mm_register, batch_packed_mm_register, deq_register

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
                        #sch = apply_parallel(sch, gv.name_hint)
                        mod[gv] = sch.mod["main"]
                    if "matmul" in gv.name_hint and isinstance(func, tir.PrimFunc) :    
                        sch_mod = tvm.IRModule({"main": func})
                        sch = tir.Schedule(sch_mod)
                        sch = apply_to_matmul(sch, gv.name_hint)
                        mod[gv] = sch.mod["main"]
        return mod
        
def apply_parallel(sch: tir.Schedule, prefill):
    attn_blk = sch.get_block("attn")
    fused_blk = sch.fuse(*sch.get_loops(attn_blk)[-2:])
    #sch.parallel(fused_blk)
    logger.info(
        "[Tensorize] Function `%s` : %s",
        prefill,
        "Parallel"
    )
    return sch

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
    sch.annotate(sch.get_block("S_gemm_o"), "pragma_import_llvm", get_ll_code(ker_name))
    return sch

def apply_to_Ogemm(sch: tir.Schedule, prefill):
    s_blk = sch.get_block("O_gemm")
    block = sch.get(s_blk)
    F_buffer = block.reads[1].buffer
    V_buffer = block.reads[2].buffer
    m_size, k_size = F_buffer.shape  # factor = [m, k]
    nk_size, n_size = V_buffer.shape # V = [k, n]
    ker_name = ogemm_register(m_size, n_size, k_size)
    # tensorize loop_x
    loop_x, loop_y, loop_z = sch.get_loops(s_blk)[-3:]
    sch.tensorize(loop_x, ker_name)
    logger.info(
        "[Tensorize] Function `%s` : %s",
        prefill,
        ker_name
    )
    sch.annotate(sch.get_block("O_gemm_o"), "pragma_import_llvm", get_ll_code(ker_name))
    return sch

def apply_to_matmul(sch: tir.Schedule, matmul):
    
    mm_blk = sch.get_block("NT_matmul")
    looprvs = sch.get_loops(mm_blk)
    if len(looprvs)  == 4:
        batch = sch.get(looprvs[0]).extent
        seqlen = sch.get(looprvs[1]).extent
        N = sch.get(looprvs[2]).extent
        K = sch.get(looprvs[3]).extent
        if not isinstance(N, tvm.tir.expr.Var):
            # DEQ and Packing
            if "matmul5" in str(matmul) or "matmul7" in str(matmul) or "matmul6" in str(matmul) or "matmul8" in str(matmul): 
                try:
                    tile = 32
                    deq_kername = deq_register(K, tile)
                    sch.transform_layout(block=sch.get_block("compute"), buffer=("write", 0), index_map=lambda n, k: [ n // tile , k , n % tile ])
                    sch.transform_layout(block=sch.get_block("dequantize"), buffer=("write", 0), index_map=lambda n, k: [ n // tile , k , n % tile ])
                    sch.compute_inline("compute")
                    d_blk = sch.get_block("dequantize")
                    i, j = sch.get_loops(d_blk)
                    io, i = sch.split(i, [None, tile])
                    jo, j = sch.split(j, [None, 32])
                    sch.reorder(io, jo, i, j)
                    sch.tensorize(i, deq_kername)

                    #sch.parallel(io)

                    logger.info(
                        "[Tensorize] Function `%s` : %s",
                        matmul,
                        deq_kername
                    )
                    sch.annotate(sch.get_block("dequantize_o"), "pragma_import_llvm", get_ll_code(deq_kername))

                    if batch == 1:
                        mm_kername = seq_packed_mm_register(4, 32, 64, K, N)
                    else:
                        mm_kername = batch_packed_mm_register(4, 32, 64, K, N)
                    mmblk = sch.get_block("NT_matmul")
                    b, i, j, k = sch.get_loops(mmblk)
                    sch.decompose_reduction(mmblk, b)
                    mmblk = sch.get_block("NT_matmul_update")
                    #b, i, j, k = sch.get_loops(mmblk)
                    i = sch.fuse(b, i)
                    io, i = sch.split(i, [None, 4], disable_predication=True)
                    jo, j = sch.split(j, [None, 32])
                    ko, k = sch.split(k, [None, 64])
                    sch.reorder(io, jo, ko, i, j, k)
                    sch.tensorize(i, mm_kername)
                    n_mod = inject_dyn_pass(sch.mod)
                    sch = tir.Schedule(n_mod)

                    #sch.parallel(io)
                    
                    sch.annotate(sch.get_block("NT_matmul_update_o"), "pragma_import_llvm", get_ll_code(mm_kername))
                    logger.info(
                        "[Tensorize] Function `%s` : %s",
                        matmul,
                        mm_kername
                    )
                except:
                    print(batch, seqlen, N, K)
            else:
                try: 
                    dyn_dim = sch.fuse(*sch.get_loops(mm_blk)[0:2])
                    loop_y = looprvs[2]
                    loop_z = looprvs[3]
                    if batch == 1: #seq_len is dynamic 
                        ker_name = seq_mmt_register(1, 2, 64, K, N)
                    else: # batch is dynamic 
                        ker_name = batch_mmt_register(1, 2, 64, K, N)
                    j_o, j, j_i = sch.split(loop_y,[None, 8, 2])
                    k_o, k_i = sch.split(loop_z,[None, 64])
                    sch.reorder(j_o, dyn_dim, j, k_o, j_i, k_i)
                    sch.tensorize(j_i, ker_name)
                    sch.annotate(sch.get_block("NT_matmul_o"), "pragma_import_llvm", get_ll_code(ker_name))


                    deq = sch.get_block("dequantize")
                    sch.reverse_compute_inline(deq)
                    de_o, de_i = sch.get_loops(sch.get_block("compute"))
                    # sch.parallel(dyn_dim)
                    # sch.parallel(de_o)
                    logger.info(
                        "[Tensorize] Function `%s` : %s",
                        matmul,
                        ker_name
                    )
                    return sch
                except:
                    print(batch, seqlen, N, K)
    return sch