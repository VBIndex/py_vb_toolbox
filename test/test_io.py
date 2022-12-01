import unittest
from unittest import TestCase
from unittest.mock import PropertyMock, patch, call, Mock, MagicMock
from vb_toolbox import io

import numpy as np


class GiftiIO_TestCase(TestCase):

    @patch('vb_toolbox.io.nibabel', spec=True)
    def test_open_gifti_surf(self, mock_nibabel):
        filename = Mock()
        data1 = PropertyMock()
        data2 = PropertyMock()
        mock_nibabel.load.return_value.get_arrays_from_intent.side_effect = [[Mock(data=data1)], [Mock(data=data2)]]
        ret = io.open_gifti_surf(filename)
        mock_nibabel.load.assert_called_with(filename)
        mock_nibabel.load.return_value.get_arrays_from_intent.assert_has_calls([call('pointset'), call('triangle')])
        self.assertEqual(
            ret, (mock_nibabel.load.return_value, data1, data2)
        )

    @patch('vb_toolbox.io.nibabel', spec=True)
    def test_open_gifti(self, mock_nibabel):
        filename = Mock()
        data1 = PropertyMock()
        mock_nibabel.load.return_value.darrays = [Mock(data=data1)] 
        ret = io.open_gifti(filename)
        mock_nibabel.load.assert_called_with(filename)
        self.assertEqual(
            ret, (mock_nibabel.load.return_value, data1)
        )

    @patch('vb_toolbox.io.nibabel', spec=True)
    @patch('vb_toolbox.io.np.array', spec=True)
    def test_save_gifti(self, mock_array, mock_nibabel):

        # with AnatomicalStructurePrimary in og_img.meta
        filename = Mock()
        anatomicalStructurePrimary = Mock()
        og_img = Mock(meta={'AnatomicalStructurePrimary': anatomicalStructurePrimary})
        data = Mock()
        io.save_gifti(og_img, data, filename)
        mock_nibabel.gifti.gifti.GiftiDataArray.assert_called_once_with(mock_array.return_value)
        mock_nibabel.gifti.gifti.GiftiMetaData.assert_called_once_with(AnatomicalStructurePrimary=anatomicalStructurePrimary)
        mock_nibabel.gifti.gifti.GiftiImage.assert_called_once_with(
            darrays=[mock_nibabel.gifti.gifti.GiftiDataArray.return_value],
            meta=mock_nibabel.gifti.gifti.GiftiMetaData.return_value
        )
        mock_nibabel.save(mock_nibabel.gifti.gifti.GiftiImage.return_value, filename)
        mock_nibabel.reset_mock()
        
        # without AnatomicalStructurePrimary in og_img.meta
        filename = Mock()
        anatomicalStructurePrimary = Mock()
        og_img = Mock(meta={})
        data = Mock()
        io.save_gifti(og_img, data, filename)
        mock_nibabel.gifti.gifti.GiftiDataArray.assert_called_once_with(mock_array.return_value)
        mock_nibabel.gifti.gifti.GiftiMetaData.assert_called_once_with()
        mock_nibabel.gifti.gifti.GiftiImage.assert_called_once_with(
            darrays=[mock_nibabel.gifti.gifti.GiftiDataArray.return_value],
            meta=mock_nibabel.gifti.gifti.GiftiMetaData.return_value
        )
        mock_nibabel.save(mock_nibabel.gifti.gifti.GiftiImage.return_value, filename)

        
if __name__ == "__main__":
    unittest.main()
