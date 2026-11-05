from sklearn.linear_model import LinearRegression


def train(X_train, y_train, config):
    """Train a Linear Regression model."""

    model = LinearRegression(**config)

    model.fit(X_train, y_train)

    return model


def predict(model, X):
    """Generate predictions."""

    return model.predict(X)