#ifndef MMAP_CACHE_H
#define MMAP_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <doca_mmap.h>

#include "../common/doca_rdma_utils.h"

/* Remote mmap キャッシュ (ハッシュ直接マッピング)
 *   ホスト側バッファの (addr, len) → import 済み doca_remote_mem_t */
#define RMEM_CACHE_SIZE     256            /* 2 のべき乗 */
#define RMEM_CACHE_MASK     (RMEM_CACHE_SIZE - 1)

struct rmem_cache_entry {
    uint64_t addr;                         /* remote_addr (GPU アドレス) */
    size_t   len;                          /* remote_len */
    struct doca_remote_mem_t rmem;
    bool     valid;
};

struct rmem_cache_entry *rmem_cache_find(
    struct rmem_cache_entry *cache, uint64_t addr, size_t len);
void rmem_cache_store(
    struct rmem_cache_entry *cache, uint64_t addr, size_t len,
    const struct doca_remote_mem_t *rmem);
void rmem_cache_destroy(struct rmem_cache_entry *cache);

/* PCI import mmap キャッシュ (GPU Direct 用, 線形探索付きハッシュ表)
 *   GPU バッファの (addr, len) → PCI import 済み doca_mmap */
#define PCI_MMAP_CACHE_SIZE   (1 << 20)         /* 1048576 = 2^20 */
#define PCI_MMAP_CACHE_MASK   (PCI_MMAP_CACHE_SIZE - 1)

struct pci_mmap_cache_entry {
    uint64_t addr;                         /* GPU アドレス */
    size_t   len;                          /* バッファサイズ */
    struct doca_mmap *mmap;                /* PCI import 済み mmap */
    bool     valid;
};

struct pci_mmap_cache_entry *pci_mmap_cache_alloc(void);
struct doca_mmap *pci_mmap_cache_find(
    struct pci_mmap_cache_entry *cache, uint64_t addr, size_t len);
void pci_mmap_cache_store(
    struct pci_mmap_cache_entry *cache, uint64_t addr, size_t len,
    struct doca_mmap *mmap);
void pci_mmap_cache_destroy(struct pci_mmap_cache_entry *cache);

#endif /* MMAP_CACHE_H */
