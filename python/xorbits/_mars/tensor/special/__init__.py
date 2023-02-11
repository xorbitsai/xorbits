# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import scipy

    from .airy import TensorAiry, TensorAirye, TensorItairy, airy, airye, itairy
    from .bessel import (
        TensorHankel1,
        TensorHankel1e,
        TensorHankel2,
        TensorHankel2e,
        TensorIV,
        TensorIVE,
        TensorJV,
        TensorJVE,
        TensorKN,
        TensorKV,
        TensorKVE,
        TensorYN,
        TensorYV,
        TensorYVE,
        hankel1,
        hankel1e,
        hankel2,
        hankel2e,
        iv,
        ive,
        jv,
        jve,
        kn,
        kv,
        kve,
        yn,
        yv,
        yve,
    )
    from .convenience import TensorXLogY, xlogy
    from .ellip_func_integrals import (
        TensorEllipe,
        TensorEllipeinc,
        TensorEllipk,
        TensorEllipkinc,
        TensorEllipkm1,
        TensorElliprc,
        TensorElliprd,
        TensorElliprf,
        TensorElliprg,
        TensorElliprj,
        ellipe,
        ellipeinc,
        ellipk,
        ellipkinc,
        ellipkm1,
        elliprc,
        elliprd,
        elliprf,
        elliprg,
        elliprj,
    )
    from .ellip_harm import (
        TensorEllipHarm,
        TensorEllipHarm2,
        TensorEllipNormal,
        ellip_harm,
        ellip_harm_2,
        ellip_normal,
    )
    from .err_fresnel import (
        TensorDawsn,
        TensorErf,
        TensorErfc,
        TensorErfcinv,
        TensorErfcx,
        TensorErfi,
        TensorErfinv,
        TensorFresnel,
        TensorModFresnelM,
        TensorModFresnelP,
        TensorVoigtProfile,
        TensorWofz,
        dawsn,
        erf,
        erfc,
        erfcinv,
        erfcx,
        erfi,
        erfinv,
        fresnel,
        modfresnelm,
        modfresnelp,
        voigt_profile,
        wofz,
    )
    from .gamma_funcs import (
        TensorBeta,
        TensorBetaInc,
        TensorBetaIncInv,
        TensorBetaLn,
        TensorDiGamma,
        TensorGamma,
        TensorGammaInc,
        TensorGammaIncc,
        TensorGammaInccInv,
        TensorGammaIncInv,
        TensorGammaln,
        TensorGammaSgn,
        TensorLogGamma,
        TensorMultiGammaLn,
        TensorPoch,
        TensorPolyGamma,
        TensorPsi,
        TensorRGamma,
        beta,
        betainc,
        betaincinv,
        betaln,
        digamma,
        gamma,
        gammainc,
        gammaincc,
        gammainccinv,
        gammaincinv,
        gammaln,
        gammasgn,
        loggamma,
        multigammaln,
        poch,
        polygamma,
        psi,
        rgamma,
    )
    from .hypergeometric_funcs import (
        TensorHYP0F1,
        TensorHYP1F1,
        TensorHYP2F1,
        TensorHYPERU,
        hyp0f1,
        hyp1f1,
        hyp2f1,
        hyperu,
    )
    from .info_theory import (
        TensorEntr,
        TensorKlDiv,
        TensorRelEntr,
        entr,
        kl_div,
        rel_entr,
    )
except ImportError:  # pragma: no cover
    pass

_names_to_del = [_name for _name, _val in globals().items() if _val is None]
[globals().pop(_name) for _name in _names_to_del]
del _names_to_del
