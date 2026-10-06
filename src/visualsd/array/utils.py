#!/usr/bin/env python3
# coding=utf-8

import os
import copy
import numpy as np

def array_transfunc(posx, posy, slomax, sloinc, fmin, fmax, finc):
    
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    n_sta = len(posx)
    
    # Evita división repetida en el bucle
    inv_nsta_sq = 1.0 / (n_sta * n_sta)
    
    freqs = np.arange(fmin, fmax + finc * 0.5, finc)
    omegas = 2.0 * np.pi * freqs
    s_vals = np.arange(-slomax, slomax + sloinc * 0.5, sloinc)
    n_s = len(s_vals)
    
    SX, SY = np.meshgrid(s_vals, s_vals, indexing='ij')
    slomax2 = slomax ** 2
    valid_mask = (SX**2 + SY**2) <= slomax2
    
    sx_valid = SX[valid_mask]
    sy_valid = SY[valid_mask]
    n_valid = len(sx_valid)
    
    power_valid = np.zeros(n_valid, dtype=np.float64)
    
    # Bucle sobre frecuencias
    for w in omegas:
        delay = sx_valid[:, None] * posx + sy_valid[:, None] * posy
        beam_sum = np.sum(np.exp(1j * w * delay), axis=1)
        power_valid += np.abs(beam_sum) ** 2 * inv_nsta_sq
        
    power_valid /= len(omegas)
    
    # Reconstruir matriz
    power = np.zeros((n_s, n_s), dtype=np.float64)
    power[valid_mask] = power_valid
    
    return s_vals, power


def delay_matrix(sx, sy, posx, posy, is_samples=False, fs=100.0):
    
    """
    Construye la matriz antisimétrica de delays (en segundos) a partir de un vector de lentitud.

    :param sx, sy: Lentitud (s/km).
    :param posx, posy: Arrays 1D con las coordenadas de cada estación (km).
    :param is_samples: Booleano. True si deseas muestras
    :param fs: Frecuencia de muestreo
    """
    
    t_abs = (sx * posx) + (sy * posy)
    delay = t_abs[:, np.newaxis] - t_abs[np.newaxis, :]

    if is_samples:
        delay /= fs

    return delay


def stack_traces(data, delay, fs, ref_index=0, normalize_product=True):
    """
    Calcula el Linear y Product Beam usando operaciones puras de matrices.
    
    :param data: Array 2D (M, N) con N trazas y M muestras temporales.
    :param delay: Matriz 2D antisimétrica (N, N) de desfases en segundos.
    :param fs: Frecuencia de muestreo en Hz
    :param ref_index: Índice de la fila que servirá como ancla (default 0).
    :param normalize_product: Normaliza antes de multiplicar para evitar under/overflow.
    
    :return: (beam_stacked, beam_product) - Arrays 1D de longitud M.
    """

    M, N = data.shape
    aligned_data = np.zeros_like(data)
    
    for i in range(N):
        # Desfase de la traza 'i' respecto a la referencia
        shift_sec = -delay[i, ref_index]
        n_shift = int(round(shift_sec * fs))
        row_data = data[:, i]
        
        # Alinear usando slicing en el eje M
        if n_shift > 0:    # Retrasar
            length = min(M, M - n_shift)
            if length > 0:
                aligned_data[n_shift : n_shift + length, i] = row_data[:length]
        
        elif n_shift < 0:  # Adelantar
            length = min(M + n_shift, M)
            if length > 0:
                aligned_data[:length, i] = row_data[-n_shift : -n_shift + length]
        
        else:
            aligned_data[:, i] = row_data
    
    beam_stacked = np.mean(aligned_data, axis=1)
    
    if normalize_product:
        # Normalización robusta
        max_vals = np.max(np.abs(aligned_data), axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        norm_data = aligned_data / max_vals

        sign_prod = np.prod(np.sign(norm_data), axis=1)
        log_abs_sum = np.sum(np.log(np.abs(norm_data) + 1e-15), axis=1)
        beam_product = sign_prod * np.exp(log_abs_sum)
    else:
        beam_product = np.prod(aligned_data, axis=1)

    return aligned_data, beam_stacked, beam_product
