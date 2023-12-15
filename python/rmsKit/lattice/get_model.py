import lattice
from typing import List, Tuple, Any, Dict, Union
from numpy._typing import NDArray


def get_model(model: str, params: Dict[str, Any], L: Union[List[int], None] = None
              ) -> Tuple[Union[List[NDArray[Any]], NDArray[Any]], int, str]:

    if "FF" in model:
        lt = params["lt"]
        sps = int(params["sps"])
        seed = params["seed"]
        if model == "FF1D":
            d = 1
        elif model == "FF2D":
            d = 2
        p = dict(
            sps=sps,
            rank=2,
            dimension=d,
            # lattice type (number of sites in unit cell)
            lt=1 if lt == "original" else int(lt),
            seed=1 if seed is None else seed,
        )
        if L:
            raise NotImplementedError("get system hamiltonian not implemented")
        h_list, sps = lattice.FF.local(p)
        # h_list: List[NDArray[Any]] = [h.numpy() for h in _h_list]
        params_str = f's_{sps}_r_{p["rank"]}_d_{p["dimension"]}_seed_{p["seed"]}'
        model_name = f"{model}_loc/{params_str}"
        return h_list, sps, model_name
    elif "HXYZ" in model:

        if model == "HXYZ1D":
            d = 1
        elif model == "HXYZ2D":
            d = 2
        p = dict(
            Jx=params["Jx"],
            Jy=params["Jy"],
            Jz=params["Jz"],
            hx=params["hx"],
            hz=params["hz"],
        )
        a = ""
        for k, v in p.items():
            v = float(v)
            a += f"{k}_{v:.4g}_"
        params_str = a[:-1]
        p["lt"] = params["lt"],

        if L:
            size_name = f"L_{L[0]}x{L[1]}" if d == 2 else f"L_{L[0]}"
            model_name = f"{model}_sys/{size_name}/{params_str}"
            h_list, sps = lattice.HXYZ.system(L, p)
            return h_list, sps, model_name
        else:
            model_name = f"{model}_loc/{params_str}"
            h_list, sps = lattice.HXYZ.local(params, d)
            return h_list, sps, model_name

    elif "BLBQ1D" in model:
        d = 1
        p = dict(
            J0=params["J0"],
            J1=params["J1"],
            hx=params["hx"],
            hz=params["hz"],
        )
        a = ""
        for k, v in p.items():
            v = float(v)
            a += f"{k}_{v:.4g}_"
        params_str = a[:-1]
        p["lt"] = params["lt"],

        if L:
            size_name = f"L_{L[0]}"
            model_name = f"{model}_sys/{size_name}/{params_str}"
            h_list, sps = lattice.BLBQ.system(L, p)
            return h_list, sps, model_name
        else:
            model_name = f"{model}_loc/{params_str}"
            h_list, sps = lattice.BLBQ.local(params, d)
            return h_list, sps, model_name
    raise NotImplementedError(f"model {model} not implemented")
