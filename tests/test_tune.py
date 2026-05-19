import pytest
from unittest.mock import patch, MagicMock
from src.tune import parse_args, main


def test_parse_args():
    with patch("sys.argv", ["tune.py", "--symbol", "VALE3.SA", "--n-trials", "2", "--max-epochs", "5"]):
        args = parse_args()
        assert args.symbol == "VALE3.SA"
        assert args.n_trials == 2
        assert args.max_epochs == 5


@patch("src.tune.run_training_pipeline")
def test_tune_main_flow(mock_run_pipeline):
    # Mock run_training_pipeline para retornar métricas fictícias
    mock_run_pipeline.return_value = {
        "metrics": {
            "lstm_val": {"mape_pct": 2.5}
        },
        "history": {},
        "output_dir": "dummy"
    }

    with patch("sys.argv", ["tune.py", "--symbol", "TEST.SA", "--n-trials", "2", "--max-epochs", "2"]):
        main()
        
    assert mock_run_pipeline.call_count == 2
