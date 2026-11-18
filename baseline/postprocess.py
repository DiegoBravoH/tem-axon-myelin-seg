"""
Semántica -> instancias en el formato del challenge (baseline ASTIH / AxonDeepSeg).

1. Componentes conexas del axón -> una semilla por axón.
2. Watershed sobre la distancia (negada) al borde del axón, restringido a axón ∪ mielina:
   cada píxel de mielina se asigna a la fibra cuyo axón "inunda" primero.
3. Renumeración: fibra k (1..N) -> axón 2k, mielina 2k+1  (uint16).

Entrada semántica: 0 fondo, 1 axón, 2 mielina (como nnU-Net; cualquier otro valor = fondo).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

AXON, MYELIN = 1, 2
CONN8 = np.ones((3, 3), dtype=bool)


def reweight_probs(p, myelin_weight: float = 1.0):
    """Multiplica P(mielina) (canal 2 de p, tensor torch o array (C, H, W)) antes del argmax.

    < 1 hace a la red más conservadora con la mielina: en el fold 0, 0,85 bajó el HD95 de 606 a 590 nm
    (Dice 0,8857 -> 0,8862, MAE_g 0,0100 -> 0,0096); 0,6-0,85 se comporta igual. 1,0 = sin cambio.
    """
    if myelin_weight != 1.0:
        p[MYELIN] = p[MYELIN] * myelin_weight
    return p


def semantic_to_instances(sem: np.ndarray, min_axon_px: int = 0) -> np.ndarray:
    axon = sem == AXON
    fiber = axon | (sem == MYELIN)
    seeds, n = ndi.label(axon, structure=CONN8)
    if min_axon_px > 0 and n:
        sizes = np.bincount(seeds.ravel())
        small = sizes < min_axon_px
        small[0] = False
        seeds[small[seeds]] = 0
        seeds, n = ndi.label(seeds > 0, structure=CONN8)
    out = np.zeros(sem.shape, dtype=np.uint16)
    if n == 0:
        return out
    if 2 * n + 1 > np.iinfo(np.uint16).max:
        raise ValueError(f"demasiadas fibras ({n}) para uint16")
    # distancia desde el exterior del axón: con ella la mielina crece desde el axón más cercano
    dist = ndi.distance_transform_edt(~axon)
    fibers = watershed(dist, markers=seeds, mask=fiber)
    out[fiber & axon] = 2 * fibers[fiber & axon]
    my = fiber & ~axon & (fibers > 0)
    out[my] = 2 * fibers[my] + 1
    return out


def fiber_scores(inst: np.ndarray, fg_prob: np.ndarray) -> np.ndarray:
    """Confianza por fibra: media de P(axón)+P(mielina) en sus píxeles. Índice = id de fibra (X // 2)."""
    fid = np.where(inst >= 2, inst // 2, 0).astype(np.int64)
    n = int(fid.max()) + 1
    area = np.bincount(fid.ravel(), minlength=n)
    tot = np.bincount(fid.ravel(), weights=fg_prob.ravel().astype(np.float64), minlength=n)
    return np.divide(tot, area, out=np.zeros(n), where=area > 0)


def filter_fibers(inst: np.ndarray, border_margin: int | None = 0, drop_axon_only: bool = False,
                  min_fiber_px: int = 0, fg_prob: np.ndarray | None = None, min_conf: float = 0.0,
                  border_mode: str = "axon", min_solidity: float = 0.0, min_circularity: float = 0.0) -> np.ndarray:
    """Elimina fibras que el GT del challenge no etiqueta como fibra (y renumera 2k / 2k+1).

    border_margin: fibras con algún píxel a <= margin px del borde de la imagen se borran
                   (en el GT, las fibras cortadas por el borde son artefacto = 1, ignorado; si se
                   predicen, lo que sobresale de la zona ignorada cuenta como falso positivo).
                   None desactiva la regla.
    border_mode:   "axon"  -> solo se borra si el AXÓN toca el borde (criterio del GT: en el fold 0,
                              60/64 fibras de borde etiquetadas tienen el axón dentro y 730/787
                              fibras marcadas como artefacto tienen el axón en el borde)
                   "fiber" -> se borra si cualquier píxel de la fibra toca el borde
    drop_axon_only: borra fibras sin mielina (g = 1; casi no existen en el GT).
    min_fiber_px:  borra fibras (axón ∪ mielina) más pequeñas que esto.
    fg_prob, min_conf: borra fibras cuya confianza media (fiber_scores) sea < min_conf.
    min_solidity / min_circularity: borra fibras (axón ∪ mielina) de forma muy irregular. En el GT
                   (8015 fibras) solo el 0,27 % tiene solidez < 0,85 y el 0,19 % circularidad < 0,5.
    """
    fid = np.where(inst >= 2, inst // 2, 0).astype(np.int64)
    n = int(fid.max()) + 1
    if n == 1:
        return inst
    remove = np.zeros(n, dtype=bool)
    if border_margin is not None:
        m = border_margin + 1
        src = np.where((inst >= 2) & (inst % 2 == 0), fid, 0) if border_mode == "axon" else fid
        edge = np.concatenate([src[:m].ravel(), src[-m:].ravel(), src[:, :m].ravel(), src[:, -m:].ravel()])
        remove[np.unique(edge)] = True
    area = np.bincount(fid.ravel(), minlength=n)
    if drop_axon_only:
        my_area = np.bincount(fid[(inst >= 3) & (inst % 2 == 1)], minlength=n)
        remove |= (area > 0) & (my_area == 0)
    if min_fiber_px > 0:
        remove |= area < min_fiber_px
    if fg_prob is not None and min_conf > 0:
        remove |= fiber_scores(inst, fg_prob) < min_conf
    if min_solidity > 0 or min_circularity > 0:
        from skimage.measure import regionprops
        for rp in regionprops(fid):
            if remove[rp.label]:
                continue
            circ = 4 * np.pi * rp.area / max(rp.perimeter, 1) ** 2
            if rp.solidity < min_solidity or circ < min_circularity:
                remove[rp.label] = True
    remove[0] = True
    keep = np.flatnonzero(~remove & (area > 0))
    new_id = np.zeros(n, dtype=np.int64)
    new_id[keep] = np.arange(1, len(keep) + 1)
    k = new_id[fid]
    out = np.where(k > 0, 2 * k + (inst % 2 == 1), 0).astype(np.uint16)
    return out
