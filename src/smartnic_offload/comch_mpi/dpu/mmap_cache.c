/* Remote mmap / PCI import mmap のキャッシュ実装。
 * ホスト・GPU バッファの import 結果を (addr, len) をキーに再利用する。 */

#include <stdlib.h>

#include "mmap_cache.h"

/* ---- Remote mmap キャッシュ ---- */

static inline uint64_t rmem_cache_hash(uint64_t addr, size_t len)
{
    uint64_t x = addr;
    x ^= (uint64_t)len + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return x & RMEM_CACHE_MASK;
}

struct rmem_cache_entry *rmem_cache_find(
    struct rmem_cache_entry *cache, uint64_t addr, size_t len)
{
    int idx = (int)rmem_cache_hash(addr, len);
    if (cache[idx].valid && cache[idx].addr == addr && cache[idx].len == len)
        return &cache[idx];
    return NULL;
}

void rmem_cache_store(
    struct rmem_cache_entry *cache, uint64_t addr, size_t len,
    const struct doca_remote_mem_t *rmem)
{
    int idx = (int)rmem_cache_hash(addr, len);
    if (cache[idx].valid)
        doca_remote_mem_destroy(&cache[idx].rmem);
    cache[idx].addr  = addr;
    cache[idx].len   = len;
    cache[idx].rmem  = *rmem;
    cache[idx].valid = true;
}

void rmem_cache_destroy(struct rmem_cache_entry *cache)
{
    for (int i = 0; i < RMEM_CACHE_SIZE; i++) {
        if (cache[i].valid) {
            doca_remote_mem_destroy(&cache[i].rmem);
            cache[i].valid = false;
        }
    }
}

/* ---- PCI import mmap キャッシュ ---- */

static inline uint64_t pci_mmap_cache_hash(uint64_t addr, size_t len)
{
    uint64_t x = addr;
    x ^= (uint64_t)len + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return x & PCI_MMAP_CACHE_MASK;
}

struct pci_mmap_cache_entry *pci_mmap_cache_alloc(void)
{
    struct pci_mmap_cache_entry *cache = (struct pci_mmap_cache_entry *)calloc(
        PCI_MMAP_CACHE_SIZE, sizeof(struct pci_mmap_cache_entry));
    return cache;
}

/* 線形探索の最大ステップ数 (無限ループ防止) */
#define PCI_MMAP_CACHE_MAX_PROBE  64

struct doca_mmap *pci_mmap_cache_find(
    struct pci_mmap_cache_entry *cache, uint64_t addr, size_t len)
{
    if (!cache) return NULL;
    int idx = (int)pci_mmap_cache_hash(addr, len);
    for (int probe = 0; probe < PCI_MMAP_CACHE_MAX_PROBE; probe++) {
        int i = (idx + probe) & PCI_MMAP_CACHE_MASK;
        if (!cache[i].valid)
            return NULL;  /* 空きスロットに到達 → 存在しない */
        if (cache[i].addr == addr && cache[i].len == len)
            return cache[i].mmap;
    }
    return NULL;
}

void pci_mmap_cache_store(
    struct pci_mmap_cache_entry *cache, uint64_t addr, size_t len,
    struct doca_mmap *mmap)
{
    if (!cache) return;
    int idx = (int)pci_mmap_cache_hash(addr, len);
    for (int probe = 0; probe < PCI_MMAP_CACHE_MAX_PROBE; probe++) {
        int i = (idx + probe) & PCI_MMAP_CACHE_MASK;
        if (!cache[i].valid) {
            /* 空きスロット発見 → 格納 */
            cache[i].addr  = addr;
            cache[i].len   = len;
            cache[i].mmap  = mmap;
            cache[i].valid = true;
            return;
        }
        if (cache[i].addr == addr && cache[i].len == len) {
            /* 同じキーが既に存在 → 更新不要 */
            return;
        }
    }
    /* キャッシュ満杯 — 呼び出し側は新しい mmap で動作するが次回 re-import が発生する */
}

void pci_mmap_cache_destroy(struct pci_mmap_cache_entry *cache)
{
    if (!cache) return;
    for (int i = 0; i < PCI_MMAP_CACHE_SIZE; i++) {
        if (cache[i].valid && cache[i].mmap) {
            doca_mmap_destroy(cache[i].mmap);
            cache[i].valid = false;
        }
    }
    free(cache);
}
