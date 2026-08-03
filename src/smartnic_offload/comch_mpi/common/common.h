/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <stdbool.h>

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_mmap.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function to check if a given device is capable of executing some task */
typedef doca_error_t (*tasks_check)(struct doca_devinfo *);

/* DOCA core objects used by the samples / applications */
typedef doca_error_t (*open_dev_cb)(struct doca_devinfo *devinfo, void *usr_ctx, struct doca_dev **dev);

/*
 * Open a DOCA device according to a given PCI address
 *
 * @pci_addr [in]: PCI address
 * @func [in]: pointer to a function that checks if the device have some task capabilities (Ignored if set to NULL)
 * @retval [out]: pointer to doca_dev struct, NULL if not found
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t open_doca_device_with_pci(const char *pci_addr, tasks_check func, struct doca_dev **retval);

/*
 * Open a DOCA device according to a given PCI address and a callback function
 *
 * @pci_addr [in]: PCI address
 * @func [in]: pointer to a function that checks if the device have some task capabilities (Ignored if set to NULL)
 * @open_dev_cb [in]: pointer to a function that opens the device
 * @usr_ctx [in]: user context
 * @retval [out]: pointer to doca_dev struct, NULL if not found
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t open_doca_device_with_pci_and_callback(const char *pci_addr,
						    tasks_check func,
						    open_dev_cb open_dev_cb,
						    void *usr_ctx,
						    struct doca_dev **retval);

/*
 * Open a DOCA device according to a given interface name
 *
 * @value [in]: interface name
 * @val_size [in]: input length, in bytes
 * @func [in]: pointer to a function that checks if the device have some task capabilities (Ignored if set to NULL)
 * @retval [out]: pointer to doca_dev struct, NULL if not found
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t open_doca_device_with_iface_name(const uint8_t *value,
					      size_t val_size,
					      tasks_check func,
					      struct doca_dev **retval);

/*
 * Open a DOCA device according to a given PCI address
 *
 * @local [in]: queries representors of the given local doca device
 * @filter [in]: bitflags filter to narrow the representors in the search
 * @pci_addr [in]: PCI address
 * @retval [out]: pointer to doca_dev_rep struct, NULL if not found
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t open_doca_device_rep_with_pci(struct doca_dev *local,
					   enum doca_devinfo_rep_filter filter,
					   const char *pci_addr,
					   struct doca_dev_rep **retval);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
