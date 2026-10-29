from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class PipelineSemana7(FlowSpec):
    n_estimators = Parameter("n_estimators", default=100, type=int)

    @step
    def start(self):
        df = pd.read_csv("data/raw/dataset.csv")
        X = df.drop("target", axis=1)
        y = df["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.next(self.treinar)

    @step
    def treinar(self):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        self.next(self.avaliar)

    @step
    def avaliar(self):
        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(f"accuracy={self.accuracy:.4f}")
        self.next(self.end)

    @step
    def end(self):
        print(f"Pipeline concluido. accuracy={self.accuracy:.4f}")

if __name__ == "__main__":
    PipelineSemana7()
