from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

def build_multi_lstm_model(th, td, tw, tp):
    input_recent = Input(shape=(th, 1), name="recent_input")
    input_daily = Input(shape=(td, 1), name="daily_input")
    input_weekly = Input(shape=(tw, 1), name="weekly_input")

    lstm_recent = LSTM(64)(input_recent)
    lstm_daily = LSTM(32)(input_daily)
    lstm_weekly = LSTM(32)(input_weekly)

    output_recent = Dense(tp, activation="linear", name="output_recent")(lstm_recent)
    output_daily = Dense(tp, activation="linear", name="output_daily")(lstm_daily)
    output_weekly = Dense(tp, activation="linear", name="output_weekly")(lstm_weekly)

    merged_output = Concatenate(name="merged_output")([output_recent, output_daily, output_weekly])
    final_output = Dense(tp, activation="linear", name="final_output")(merged_output)

    model = Model(
        inputs=[input_recent, input_daily, input_weekly],
        outputs=[output_recent, output_daily, output_weekly, final_output]
    )

    model.compile(
        optimizer="adam",
        loss={
            "output_recent": "mse",
            "output_daily": "mse",
            "output_weekly": "mse",
            "final_output": "mse"
        },
        loss_weights={
            "output_recent": 0.0,
            "output_daily": 0.0,
            "output_weekly": 0.0,
            "final_output": 1.0
        },
        metrics={
            "final_output": ["mae"]
        }
    )

    return model
