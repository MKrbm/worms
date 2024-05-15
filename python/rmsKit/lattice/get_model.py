from .models import FF, HXYZ, BLBQ, MG, SS, KH
from typing import List, Tuple, Any, Dict, Union, Optional
from numpy._typing import NDArray


def get_model(model: str, params: Dict[str, Any], L: Optional[List[int]] = None
              ) -> Tuple[NDArray, int, str]:

    if "FF" in model:
        lt = params["lt"]
        sps = int(params["sps"])
        seed = params["seed"]
        d = 1
        if model == "FF1D":
            d = 1
        elif model == "FF2D":
            d = 2
        p = dict(
            sps=sps,
            rank=2,
            dimension=d,
            # type (number of sites in unit cell)
            lt=1 if lt == "original" else int(lt),
            seed=1 if seed is None else seed,
        )
        params_str = f's_{sps}_r_{p["rank"]}_d_{p["dimension"]}_seed_{p["seed"]}'
        if L:
            size_name = f"L_{L[0]}x{L[1]}" if d == 2 else f"L_{L[0]}"
            model_name = f"{model}_sys/{size_name}/{params_str}"
            h_list, sps = FF.system(L, p)
            return h_list, sps, model_name
        else:
            h_list, sps = FF.local(p)
            # h_list: List[NDArray[Any]] = [h.numpy() for h in _h_list]
            model_name = f"{model}_loc/{params_str}"
            return h_list, sps, model_name
    elif "HXYZ" in model:

        d = 1
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
        p["lt"] = params["lt"]
        p["obc"] = params["obc"]

        if L:
            size_name = f"L_{L[0]}x{L[1]}" if d == 2 else f"L_{L[0]}"
            model_name = f"{model}_sys/{size_name}/{params_str}"
            h_list, sps = HXYZ.system(L, p)
            return h_list, sps, model_name
        else:
            model_name = f"{model}_loc/{params_str}"
            h_list, sps = HXYZ.local(params, d)
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
        p["lt"] = params["lt"]
        p["obc"] = params["obc"]

        if L:
            size_name = f"L_{L[0]}"
            model_name = f"{model}_sys/{size_name}/{params_str}"
            h_list, sps = BLBQ.system(L, p)
            return h_list, sps, model_name
        else:
            model_name = f"{model}_loc/{params_str}"
            h_list, sps = BLBQ.local(params, d)
            return h_list, sps, model_name

    elif "MG1D" in model:
        d = 1
        p = dict(
            J1=params["J1"],
            J2=params["J2"],
            J3=params["J3"],
        )
        a = ""
        for k, v in p.items():
            v = float(v)
            a += f"{k}_{v:.4g}_"
        params_str = a[:-1]
        p["lt"] = params["lt"]
        p["obc"] = params["obc"]

        if L:
            size_name = f"L_{L[0]}"
            model_name = f"{model}_sys/{size_name}/{params_str}"
            h_list, sps = MG.system(L, p)
            return h_list, sps, model_name
        else:
            model_name = f"{model}_loc/{params_str}"
            h_list, sps = MG.local(params, d)
            return h_list, sps, model_name

    elif "SS2D" in model:
        p = dict(
            J0=params["J0"],
            J1=params["J1"],
            J2=params["J2"],
        )
        a = ""
        for k, v in p.items():
            v = float(v)
            a += f"{k}_{v:.4g}_"
        params_str = a[:-1]
        p["lt"] = params["lt"]

        if L:
            if len(L) != 2:
                raise ValueError("SS only accept 2D lattice")
            size_name = f"L_{L[0]}x{L[1]}"
            model_name = f"{model}_sys/{size_name}/{params_str}"
            h_list, sps = SS.system(L, p)
            return h_list, sps, model_name
        else:
            model_name = f"{model}_loc/{params_str}"
            h_list, sps = SS.local(params, 2)
            return h_list, sps, model_name

    elif "KH2D" in model:
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
        p["lt"] = params["lt"]

        if L:
            size_name = f"L_{L[0]}x{L[1]}"
            model_name = f"{model}_sys/{size_name}/{params_str}"
            h_list, sps = KH.system(L, p)
            return h_list, sps, model_name
        else:
            model_name = f"{model}_loc/{params_str}"
            h_list, sps = KH.local(params, 2)
            return h_list, sps, model_name
    raise NotImplementedError(f"model {model} not implemented")
