import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading
import sys
import io
import queue
import numpy as np
from scipy.ndimage import gaussian_filter

MAX_LOG_LINES = 100  # Set the maximum number of log lines

class Logger(io.StringIO):
    def __init__(self, log_queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_queue = log_queue

    def write(self, message):
        # Split the message into lines and put each line in the queue with the prefix
        for line in message.splitlines():
            if line.strip():
                self.log_queue.put(f">> {line}\n")

    def flush(self):
        pass

# Define a function to load the ZoeDepth model from PyTorch Hub
def load_model(app, model_name):
    def task():
        try:
            # Redirect stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = Logger(app.log_queue)
            sys.stderr = Logger(app.log_queue)

            if model_name == "ZoeDepth":
                app.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N",pretrained=True,trust_repo=True)  # trust_repo=True to avoid permission issues
            elif model_name == "MiDaS":
                app.model = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_384", trust_repo=True)

            app.model.eval()
            app.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            app.model.to(app.device)

            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Update status with final message
            app.log_queue.put(f"{model_name} model loaded successfully.\n")
        except AttributeError as e:
            app.log_queue.put(f"AttributeError loading model: {str(e)}\n")
            # Restore stdout and stderr in case of error
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception as e:
            app.log_queue.put(f"Error loading model: {str(e)}\n")
            # Restore stdout and stderr in case of error
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    threading.Thread(target=task).start()

def process_log_queue(app):
    try:
        while True:
            message = app.log_queue.get_nowait()
            app.status_text.insert(tk.END, message)
            app.status_text.see(tk.END)  # Auto-scroll to the end

            # Clear the text box if it exceeds MAX_LOG_LINES
            while int(app.status_text.index('end-1c').split('.')[0]) > MAX_LOG_LINES:
                app.status_text.delete('1.0', '2.0')
    except queue.Empty:
        pass
    app.root.after(100, process_log_queue, app)

# Define a function to load and preprocess the image
def load_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((480, 640)),  # Resize to the input size expected by the model
        transforms.ToTensor(),  # Convert the image to a tensor
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image.to(device), Image.open(image_path).convert('RGB')

# Define a function to generate the depth map
def generate_depth_map(image_tensor, model):
    with torch.no_grad():
        outputs = model(image_tensor)
    depth_map = outputs['metric_depth']  # Use the correct key for the depth map tensor
    return depth_map

# Define a function to visualize the depth map
def visualize_depth_map(depth_map, app, display_mode):
    depth_map_np = depth_map.squeeze().cpu().detach().numpy()
    depth_map_img = (depth_map_np - depth_map_np.min()) / (depth_map_np.max() - depth_map_np.min())  # Normalize to [0, 1]
    
    if display_mode == "Inverted":
        depth_map_img = 1 - depth_map_img  # Invert the depth map if inverted mode is selected
    
    depth_map_img = (depth_map_img * 255).astype(np.uint8)  # Convert to [0, 255]
    depth_map_img = Image.fromarray(depth_map_img)

    # Resize to match the display size
    fixed_width = app.image_label.winfo_width()
    fixed_height = app.image_label.winfo_height()
    depth_map_img = depth_map_img.resize((fixed_width, fixed_height), Image.Resampling.LANCZOS)

    app.depth_map_photo = ImageTk.PhotoImage(depth_map_img)
    app.depth_map_label.config(image=app.depth_map_photo)
    app.depth_map_label.image = app.depth_map_photo

# Define a function to save the depth map
def save_depth_map(depth_map, output_path, display_mode):
    depth_map_np = depth_map.squeeze().cpu().detach().numpy()
    depth_map_np = (depth_map_np - depth_map_np.min()) / (depth_map_np.max() - depth_map_np.min())  # Normalize

    if display_mode == "Inverted":
        depth_map_np = 1 - depth_map_np  # Invert if necessary

    plt.imsave(output_path, depth_map_np, cmap='gray')

# Define a function to generate higher quality depth maps
def generate_high_quality_depth_map(image, model, log_queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    
    log_queue.put("Starting low resolution depth map generation...\n")
    low_res_depth = model(image_tensor)['metric_depth'].squeeze().cpu().detach().numpy()
    low_res_scaled_depth = 2**16 - (low_res_depth - np.min(low_res_depth)) * 2**16 / (np.max(low_res_depth) - np.min(low_res_depth))
    
    low_res_depth_map_image = Image.fromarray((0.999 * low_res_scaled_depth).astype("uint16"))
    low_res_depth_map_image.save('zoe_depth_map_16bit_low.png')
    log_queue.put("Low resolution depth map generated.\n")
    
    im = np.asarray(image)
    tile_sizes = [[4, 4], [8, 8]]
    filters = []

    for tile_size in tile_sizes:
        num_x = tile_size[0]
        num_y = tile_size[1]
        M = im.shape[0] // num_x
        N = im.shape[1] // num_y

        filter_dict = {f'{direction}_filter': np.zeros((M, N)) for direction in 
                      ['right', 'left', 'top', 'bottom', 'top_right', 'top_left', 'bottom_right', 'bottom_left']}
        filter_dict['filter'] = np.zeros((M, N))

        for i in range(M):
            for j in range(N):
                x_value = 0.998 * np.cos((abs(M / 2 - i) / M) * np.pi)**2
                y_value = 0.998 * np.cos((abs(N / 2 - j) / N) * np.pi)**2

                filter_dict['right_filter'][i, j] = x_value if j > N / 2 else x_value * y_value
                filter_dict['left_filter'][i, j] = x_value if j < N / 2 else x_value * y_value
                filter_dict['top_filter'][i, j] = y_value if i < M / 2 else x_value * y_value
                filter_dict['bottom_filter'][i, j] = y_value if i > M / 2 else x_value * y_value

                filter_dict['top_right_filter'][i, j] = (0.998 if (j > N / 2 and i < M / 2) else 
                                                         x_value if j > N / 2 else 
                                                         y_value if i < M / 2 else x_value * y_value)
                
                filter_dict['top_left_filter'][i, j] = (0.998 if (j < N / 2 and i < M / 2) else 
                                                        x_value if j < N / 2 else 
                                                        y_value if i < M / 2 else x_value * y_value)

                filter_dict['bottom_right_filter'][i, j] = (0.998 if (j > N / 2 and i > M / 2) else 
                                                            x_value if j > N / 2 else 
                                                            y_value if i > M / 2 else x_value * y_value)
                
                filter_dict['bottom_left_filter'][i, j] = (0.998 if (j < N / 2 and i > M / 2) else 
                                                           x_value if j < N / 2 else 
                                                           y_value if i > M / 2 else x_value * y_value)
                
                filter_dict['filter'][i, j] = x_value * y_value

        filters.append(filter_dict)

        for filter in filter_dict:
            filter_image = Image.fromarray((filter_dict[filter] * 2**16).astype("uint16"))
            filter_image.save(f'mask_{filter}_{num_x}_{num_y}.png')

    compiled_tiles_list = []

    for i in range(len(filters)):
        num_x = tile_sizes[i][0]
        num_y = tile_sizes[i][1]
        M = im.shape[0] // num_x
        N = im.shape[1] // num_y

        compiled_tiles = np.zeros((im.shape[0], im.shape[1]))

        x_coords = list(range(0, im.shape[0], im.shape[0] // num_x))[:num_x]
        y_coords = list(range(0, im.shape[1], im.shape[1] // num_y))[:num_y]
        x_coords_between = list(range((im.shape[0] // num_x) // 2, im.shape[0], im.shape[0] // num_x))[:num_x - 1]
        y_coords_between = list(range((im.shape[1] // num_y) // 2, im.shape[1], im.shape[1] // num_y))[:num_y - 1]

        x_coords_all = x_coords + x_coords_between
        y_coords_all = y_coords + y_coords_between

        for x in x_coords_all:
            for y in y_coords_all:
                x_end = min(x + M, im.shape[0])
                y_end = min(y + N, im.shape[1])

                depth = model(transforms.ToTensor()(Image.fromarray(np.uint8(im[x:x_end, y:y_end]))).unsqueeze(0).to(device))['metric_depth'].squeeze().cpu().detach().numpy()
                scaled_depth = 2**16 - (depth - np.min(depth)) * 2**16 / (np.max(depth) - np.min(depth))

                if y == min(y_coords_all) and x == min(x_coords_all):
                    selected_filter = filters[i]['top_left_filter']
                elif y == min(y_coords_all) and x == max(x_coords_all):
                    selected_filter = filters[i]['bottom_left_filter']
                elif y == max(y_coords_all) and x == min(x_coords_all):
                    selected_filter = filters[i]['top_right_filter']
                elif y == max(y_coords_all) and x == max(x_coords_all):
                    selected_filter = filters[i]['bottom_right_filter']
                elif y == min(y_coords_all):
                    selected_filter = filters[i]['left_filter']
                elif y == max(y_coords_all):
                    selected_filter = filters[i]['right_filter']
                elif x == min(x_coords_all):
                    selected_filter = filters[i]['top_filter']
                elif x == max(x_coords_all):
                    selected_filter = filters[i]['bottom_filter']
                else:
                    selected_filter = filters[i]['filter']

                filter_slice = selected_filter[:x_end-x, :y_end-y]
                low_res_slice = low_res_scaled_depth[x:x_end, y:y_end]
                tile_shape = filter_slice.shape

                # Ensure all shapes match
                if filter_slice.shape == scaled_depth[:tile_shape[0], :tile_shape[1]].shape == low_res_slice.shape:
                    compiled_tiles[x:x_end, y:y_end] += filter_slice * (np.mean(low_res_slice) + np.std(low_res_slice) * ((scaled_depth[:tile_shape[0], :tile_shape[1]] - np.mean(scaled_depth[:tile_shape[0], :tile_shape[1]])) / np.std(scaled_depth[:tile_shape[0], :tile_shape[1]])))
                
                log_queue.put(f"Processed tile ({x}, {y}) to ({x_end}, {y_end}) for tile size {tile_size}.\n")

        compiled_tiles[compiled_tiles < 0] = 0
        compiled_tiles = np.nan_to_num(compiled_tiles)  # Replace NaN with 0 and infinity with large finite numbers
        compiled_tiles_list.append(compiled_tiles)

        max_val = np.max(compiled_tiles)
        if max_val > 0:
            compiled_tiles = 2**16 * 0.999 * compiled_tiles / max_val
        compiled_tiles = compiled_tiles.astype("uint16")
        tiled_depth_map = Image.fromarray(compiled_tiles)
        tiled_depth_map.save(f'tiled_depth_{i}.png')
        log_queue.put(f"Tiled depth map for tile size {tile_size} saved.\n")

    grey_im = np.mean(im, axis=2)
    tiles_blur = gaussian_filter(grey_im, sigma=20)
    tiles_difference = tiles_blur - grey_im
    tiles_difference = tiles_difference / np.max(tiles_difference)
    tiles_difference = gaussian_filter(tiles_difference, sigma=40)
    tiles_difference *= 5
    tiles_difference = np.clip(tiles_difference, 0, 0.999)
    
    mask_image = Image.fromarray((tiles_difference * 2**16).astype("uint16"))
    mask_image.save('mask_image.png')
    log_queue.put("Mask image saved.\n")

    # Ensure all shapes match for combination
    min_shape = np.min([compiled_tiles_list[0].shape, compiled_tiles_list[1].shape, low_res_scaled_depth.shape], axis=0)
    tiles_difference = tiles_difference[:min_shape[0], :min_shape[1]]
    combined_result = (tiles_difference * compiled_tiles_list[1][:min_shape[0], :min_shape[1]] + (1 - tiles_difference) * ((compiled_tiles_list[0][:min_shape[0], :min_shape[1]] + low_res_scaled_depth[:min_shape[0], :min_shape[1]]) / 2)) / 2
    combined_result = np.nan_to_num(combined_result)  # Replace NaN with 0 and infinity with large finite numbers

    max_val = np.max(combined_result)
    if max_val > 0:
        combined_result = 2**16 * 0.999 * combined_result / max_val
    combined_result = combined_result.astype("uint16")

    combined_image = Image.fromarray(combined_result)
    combined_image.save('combined_image_depth.png')
    log_queue.put("High quality depth map generation complete.\n")
    return combined_image

# GUI Application
class DepthMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Map Generator - Random Code Guy")

        self.model = None
        self.device = None

        # Create toolbar
        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.model_var = tk.StringVar(value="Select Model")
        self.model_selector = tk.OptionMenu(self.toolbar, self.model_var, "MiDaS", "ZoeDepth")
        self.model_selector.pack(side=tk.LEFT, padx=2, pady=2)

        self.display_mode_var = tk.StringVar(value="Normal")
        self.display_mode_selector = tk.OptionMenu(self.toolbar, self.display_mode_var, "Normal", "Inverted")
        self.display_mode_selector.pack(side=tk.LEFT, padx=2, pady=2)

        self.load_model_button = tk.Button(self.toolbar, text="Load Model", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT, padx=2, pady=2)

        self.load_button = tk.Button(self.toolbar, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=2, pady=2)

        self.generate_button = tk.Button(self.toolbar, text="Generate Depth Map", command=self.generate_depth_map)
        self.generate_button.pack(side=tk.LEFT, padx=2, pady=2)

        self.save_button = tk.Button(self.toolbar, text="Save Depth Map", command=self.save_depth_map)
        self.save_button.pack(side=tk.LEFT, padx=2, pady=2)

        # Create image display frame
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame, bg="grey")
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.depth_map_label = tk.Label(self.image_frame, bg="grey")
        self.depth_map_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create log area
        self.status_text = tk.Text(root, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)

        self.log_queue = queue.Queue()
        process_log_queue(self)

    def load_model(self):
        model_name = self.model_var.get()
        if model_name not in ["MiDaS", "ZoeDepth"]:
            messagebox.showwarning("Warning", "Please select a model.")
            return
        load_model(self, model_name)

    def load_image(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please load the model first.")
            return
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not self.image_path:
            return
        self.image_tensor, self.display_image = load_image(self.image_path, self.device)
        
        fixed_width = self.image_label.winfo_width()
        fixed_height = self.image_label.winfo_height()
        self.display_image = self.display_image.resize((fixed_width, fixed_height), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def generate_depth_map(self):
        def task():
            if not self.image_path:
                messagebox.showwarning("Warning", "Please load an image first.")
                return
            self.log_queue.put("Starting depth map generation...\n")
            display_mode = self.display_mode_var.get()
            high_quality_depth_map = generate_high_quality_depth_map(self.display_image, self.model, self.log_queue)
            high_quality_depth_map.save('high_quality_depth_map.png')
            self.depth_map = generate_depth_map(self.image_tensor, self.model)
            visualize_depth_map(self.depth_map, self, display_mode)

        threading.Thread(target=task).start()

    def save_depth_map(self):
        if self.depth_map is None:
            messagebox.showwarning("Warning", "Please generate the depth map first.")
            return
        display_mode = self.display_mode_var.get()
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not save_path:
            return
        save_depth_map(self.depth_map, save_path, display_mode)
        messagebox.showinfo("Info", "Depth map saved successfully.")

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = DepthMapApp(root)
    root.mainloop()
