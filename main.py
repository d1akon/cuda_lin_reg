from utils.config_reader import load_config
from data.loader import load_dataset
from model.trainer import LinearRegressorGPU
from visualization.plotter import Visualizer
from vispy import app
from vispy.app import Timer

def main():
    config = load_config()

    #----- Load dataset
    x_np, y_np = load_dataset(config["dataset"])

    #----- Initialize model & visualizer
    model = LinearRegressorGPU(x_np, y_np, learning_rate=config["learning_rate"])
    vis = Visualizer(x_np, y_np, epochs=config["epochs"])

    #----- Define update callback
    def update(event):
        if model.epoch >= config["epochs"]:
            print("Training finished.")
            timer.stop()
            return

        mse = model.step()
        vis.update(model)

        print(f"Epoch {model.epoch}: w={model.w.get()[0]:.5f}, b={model.b.get()[0]:.5f}, loss={mse:.6f}")

    #----- Start timer for periodic updates
    timer = Timer(interval=config["update_interval"], connect=update, start=True)

    #----- Launch VisPy app
    app.run()

if __name__ == "__main__":
    main()
