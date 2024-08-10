import pytest


def test_execution_with_process_exit_message(mocker):
    import numpy as np
    from xoscar.errors import ServerClosed

    import xorbits
    import xorbits.remote as xr

    mocker.patch(
        "xorbits._mars.services.subtask.api.SubtaskAPI.run_subtask_in_slot",
        side_effect=ServerClosed,
    )

    with pytest.raises(
        ServerClosed,
        match=r".*?\(.*?\) with address .*? Out-of-Memory \(OOM\) problem",
    ):
        xorbits.run(xr.spawn(lambda *_: np.random.rand(10**4, 10**4)))
