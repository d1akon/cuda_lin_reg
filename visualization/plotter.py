import vispy.plot as vp
from vispy.scene import Text
from vispy.scene.visuals import Markers
import numpy as np

class Visualizer:
    def __init__(self, x_np, y_np, epochs):
        self.x_np = x_np
        self.fig = vp.Fig(show=False)
        self.plot_data = self.fig[0, 0]
        self.plot_loss = self.fig[0, 1]

        #----- Real data points (blue)
        self.scatter = self.plot_data.plot(
            (x_np.copy(), y_np.copy()),
            symbol='o',
            face_color=(0.0, 0.0, 1.0, 1.0),  # Blue
            edge_color=(0.0, 0.0, 1.0, 1.0),
            width=0
        )

        #----- Regression line (red)
        self.line_data = np.stack([x_np.copy(), np.zeros_like(x_np)], axis=1)
        self.line_plot = self.plot_data.plot(self.line_data, color=(1.0, 0.0, 0.0, 1.0), width=1)

        #----- Regression prediction points (explicit red)
        self.regression_points = Markers()
        self.regression_points.set_data(
            pos=np.column_stack([x_np.copy(), np.zeros_like(x_np)]),
            face_color=(1.0, 0.0, 0.0, 1.0),   # Solid red fill
            edge_color=(0.0, 0.0, 0.0, 1.0),   # Black edge
            size=6
        )
        self.plot_data.view.add(self.regression_points)

        #----- Loss line (green)
        self.loss_line = self.plot_loss.plot(([0], [1e-6]), color='green')
        self.plot_loss.view.camera.set_range(x=(0, epochs), y=(0, 0.1))

        #----- Text overlay (top-left)
        self.text_overlay = Text(
            '',
            parent=self.plot_data.view.scene,
            color='white',
            font_size=12,
            anchor_x='left',
            anchor_y='top'
        )
        self.text_overlay.pos = self.plot_data.view.camera.rect.left, self.plot_data.view.camera.rect.top

        self.fig.show(run=False)

    def update(self, model):
        #----- Update regression line
        y_pred = model.predict()
        self.line_data[:, 1] = y_pred
        self.line_plot.set_data(self.line_data)

        #----- Update regression points with explicit style
        x_vals = model.x.get()
        positions = np.column_stack([x_vals, y_pred])
        self.regression_points.set_data(
            pos=positions,
            face_color=(1.0, 0.0, 0.0, 1.0),  # red
            edge_color=(0.0, 0.0, 0.0, 1.0),  # black
            size=6
        )

        #----- Update loss line
        loss_x = np.arange(1, len(model.losses) + 1)
        loss_y = np.array(model.losses)
        self.loss_line.set_data((loss_x, loss_y))

        #----- Update overlay text
        w_val, b_val = model.get_parameters()
        last_loss = model.losses[-1]
        self.text_overlay.text = f"Epoch: {model.epoch}\nLoss: {last_loss:.6f}\nW: {w_val:.4f}\nB: {b_val:.4f}"

        self.fig.native.update()
