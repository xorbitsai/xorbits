pytest -s --timeout=1500 \
-k "test_date_range_execution"\
            -W ignore::PendingDeprecationWarning \
            --cov-config=setup.cfg --cov-report=xml --cov=xorbits/deploy --cov=xorbits \
            xorbits/_mars/dataframe/datasource/tests/test_datasource_execution.py > pytest.log 2>&1 