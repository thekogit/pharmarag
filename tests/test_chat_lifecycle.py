import unittest
from unittest.mock import patch, MagicMock
from chat import is_server_ready

class TestChatLifecycle(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_is_server_ready_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        assert is_server_ready("http://localhost:8080") is True

    @patch("urllib.request.urlopen")
    def test_is_server_ready_failure(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        
        assert is_server_ready("http://localhost:8080") is False
