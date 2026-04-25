-- KeaBuilder ML Schema
-- Author: Suman (jsonusuman351@gmail.com)

CREATE TABLE user_inputs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL,
    input_text  TEXT NOT NULL,
    input_type  VARCHAR(50) DEFAULT 'chat',
    source      VARCHAR(50) DEFAULT 'ask_kea',
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    input_id        UUID REFERENCES user_inputs(id),
    model_name      VARCHAR(100) NOT NULL,
    model_version   VARCHAR(20) DEFAULT '1.0',
    prediction      JSONB NOT NULL,
    confidence      FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    latency_ms      INT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_inputs_user ON user_inputs(user_id);
CREATE INDEX idx_inputs_type ON user_inputs(input_type);
CREATE INDEX idx_predictions_input ON predictions(input_id);
CREATE INDEX idx_predictions_model ON predictions(model_name);
