import numpy as np
import pandas as pd
import pytest

from fabfederate.config import load_config
from fabfederate.data.uniwear import prepare_uniwear


def test_uniwear_separates_sources_and_holds_out_complete_experiments(tmp_path):
    rows = [{'dataset_tag': 'nuaa', 'experiment_tag': f'W{i}', 'timestamp': float(j),
             'force_z': i+j, 'vibration_x': i-j, 'vibration_y': i*j,
             'tool_wear': .1+j*.01} for i in range(1, 10) for j in range(5)]
    rows.append(dict(rows[0], dataset_tag='phm2010', experiment_tag='c1'))
    path=tmp_path/'input.csv'
    pd.DataFrame(rows).to_csv(path,index=False)
    config=load_config()
    config.update(dataset='uniwear_nuaa',data_path=str(path))
    result=prepare_uniwear(config,tmp_path/'prepared')
    assert len(result.frame)==45
    assert result.frame.groupby('group_id').split.nunique().max()==1
    assert set(result.feature_names)=={'force_z','vibration_x','vibration_y'}
    assert set(result.frame.dataset_id)=={'uniwear_nuaa'}
    assert 'Derived targets' in result.metadata['target_provenance']
    rows[0]['force_z']=np.inf
    pd.DataFrame(rows).to_csv(path,index=False)
    with pytest.raises(ValueError,match='finite'):
        prepare_uniwear(config,tmp_path/'invalid')
