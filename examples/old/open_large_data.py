import pandas as pd
import tensorstore as ts
import json
from copy import deepcopy

import napari
import numpy as np
import tensorstore as ts
import trackarray_tensorstore as tats
from napari_travali2._stateful_widget import StateMachineWidget
from napari_travali2._logging import logger
import numpy as np

logger.setLevel("DEBUG")
logger.info("Starting napari-travali2")

spec = {
    "driver": "zarr3",
    "kvstore": {
            "driver": "file",
            "path": "/Volumes/Extreme SSD/aligned_image5_labels.zarr"
    },
    'context': {
            'cache_pool': {
                'total_bytes_limit': 100_000_000
            }
        },
        'recheck_cached_data':False,
    }
labels = ts.open(spec).result()[:,0,:,:]
image_spec = deepcopy(spec)
image_spec['kvstore']['path'] = '/Volumes/Extreme SSD/aligned_image5_image.zarr'
image = ts.open(image_spec).result()[:,0,:,:]

viewer = napari.Viewer()
viewer.add_image([image, image[:,::2,::2]], name='image')
viewer.add_labels([labels,labels[:,::2,::2]], name='labels')
if __name__ == '__main__':
    napari.run()


#bboxes_df = pd.read_csv('/Volumes/Extreme SSD/aligned_image5_bboxes_df.csv')
#with open('/Volumes/Extreme SSD/aligned_image5_labels.splits.json') as f:
#    splits = json.load(f, object_hook=lambda d: {int(k): [int(_v) for _v in v] for k, v in d.items()})
#
#print(labels.shape)
#ta = tats.TrackArray(labels, splits, {}, bboxes_df)
#
#viewer = Viewer()
#widget = StateMachineWidget(viewer, ta, image, crop_size=2048)
#viewer.window.add_dock_widget(widget, area="right")
#viewer.dims.set_current_step(0,0)
