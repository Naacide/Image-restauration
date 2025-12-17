# -*- coding: utf-8 -*-
"""
Ma321 BE: Image Restoration Library

@author: Nathan_AZO
"""

import numpy as np
import cv2
import copy

def masque(F, v):
    """
    Create a binary mask where pixels equal to value `v` are set to 0 and others to 1.

    Parameters
    ----------
    F : np.ndarray
        Input image (grayscale or color).
    v : int
        Pixel value to mask.

    Returns
    -------
    M : np.ndarray
        Binary mask of the same height and width as `F`.
    """
    p, q = F.shape[:2]
    r = np.ndim(F)
    M = np.zeros((p, q))
    
    if r == 2:
        M[F != v] = 1
    elif r == 3:
        M[F[:, :, 0] != v] = 1

    return M

def D(r):
    """
    Create a second-order difference matrix of size (r-2, r).

    Parameters
    ----------
    r : int
        Size of the square image dimension.

    Returns
    -------
    D_matrix : np.ndarray
        Second-order difference matrix.
    """
    D_matrix = np.eye(r-2, r)
    for j in range(r):
        for i in range(r-2):
            if i + 1 == j:
                D_matrix[i, j] = -2
            elif i + 2 == j:
                D_matrix[i, j] = 1
    return D_matrix

def A(X, M):
    """
    Apply the regularization operator to the image X using mask M.

    Parameters
    ----------
    X : np.ndarray
        Image matrix.
    M : np.ndarray
        Binary mask.

    Returns
    -------
    result : np.ndarray
        Result after applying the operator.
    """
    p, q = M.shape
    Mb = 1 - M
    return (D(p).T @ D(p) @ X + X @ D(q).T @ D(q)) * Mb

def GPC(M, F, X0, epsilon):
    """
    Gradient Projection Conjugate method to restore missing pixels.

    Parameters
    ----------
    M : np.ndarray
        Mask.
    F : np.ndarray
        Original image with missing pixels.
    X0 : np.ndarray
        Initial guess image.
    epsilon : float
        Convergence threshold.

    Returns
    -------
    X : np.ndarray
        Restored image.
    """
    X = copy.copy(X0)
    d = -(A(X, M) + A(F, M))
    compteur = 0
    Y = X + epsilon * np.ones_like(X)
    
    while np.linalg.norm(X - Y) > epsilon and compteur < 100:
        Y = copy.copy(X)
        t = -(np.trace(d.T @ (A(X, M) + A(F, M)))) / (np.trace(d.T @ A(d, M)))
        X = X + t * d
        beta = (np.trace(d.T @ A((A(X, M) + A(F, M)), M))) / (np.trace(d.T @ A(d, M)))
        d = -(A(X, M) + A(F, M)) + beta * d
        compteur += 1
        
    return np.asarray(X, dtype=np.uint8)

def restauration(F, M, epsilon):
    """
    Restore a grayscale image using mask M.

    Parameters
    ----------
    F : np.ndarray
        Input grayscale image.
    M : np.ndarray
        Mask for missing pixels.
    epsilon : float
        Convergence threshold.

    Returns
    -------
    Im_finale : np.ndarray
        Restored grayscale image.
    """
    p, q = F.shape
    X0 = np.zeros((p, q))
    Mat = F + GPC(M, F, X0, epsilon)
    Im_finale = np.clip(Mat, 0, 255).astype(np.uint8)
    return Im_finale

def restauration_couleur(F, M, epsilon):
    """
    Restore a color image using mask M by processing each channel separately.

    Parameters
    ----------
    F : np.ndarray
        Input color image.
    M : np.ndarray
        Mask for missing pixels.
    epsilon : float
        Convergence threshold.

    Returns
    -------
    Im_finale : np.ndarray
        Restored color image.
    """
    Im_finale = np.zeros_like(F)
    for i in range(3):
        Im_finale[:, :, i] = restauration(F[:, :, i], M, epsilon)
    return Im_finale

def video(F, M, epsilon, output_file="output.avi"):
    """
    Restore video frame (single image) and save as video.

    Parameters
    ----------
    F : np.ndarray
        Input frame (grayscale).
    M : np.ndarray
        Mask for missing pixels.
    epsilon : float
        Convergence threshold.
    output_file : str
        Name of the output video file.
    """
    p, q = F.shape[:2]
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc, fps, (q, p), isColor=False)
    restored_frame = restauration(F, M, epsilon)
    out.write(restored_frame)
    out.release()
