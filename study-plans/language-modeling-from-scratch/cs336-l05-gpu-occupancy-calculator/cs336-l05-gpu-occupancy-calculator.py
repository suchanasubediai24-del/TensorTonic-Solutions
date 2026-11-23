
def gpu_occupancy(
    threads_per_block: int, registers_per_thread: int, shared_mem_per_block: int,
    max_threads_per_sm: int, max_warps_per_sm: int, max_blocks_per_sm: int,
    max_registers_per_sm: int, max_shared_mem_per_sm: int, warp_size: int = 32,
) -> dict:
    """
    Returns blocks_per_sm, resident_warps, and occupancy.
    """

    # Calculate warps per block using ceiling division
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size

    # Calculate effective threads per block
    effective_threads = warps_per_block * warp_size

    # Start with maximum block limit
    blocks = max_blocks_per_sm

    # Apply thread limit
    blocks = min(blocks, max_threads_per_sm // effective_threads)

    # Apply warp limit
    blocks = min(blocks, max_warps_per_sm // warps_per_block)

    # Apply register limit
    if registers_per_thread > 0:
        registers_per_block = registers_per_thread * effective_threads
        blocks = min(blocks, max_registers_per_sm // registers_per_block)

    # Apply shared memory limit
    if shared_mem_per_block > 0:
        blocks = min(blocks, max_shared_mem_per_sm // shared_mem_per_block)

    # Calculate resident warps
    resident_warps = blocks * warps_per_block

    # Calculate occupancy
    occupancy = resident_warps / max_warps_per_sm

    return {
        "blocks_per_sm": int(blocks),
        "resident_warps": int(resident_warps),
        "occupancy": float(occupancy)
    }
