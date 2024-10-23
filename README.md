### This code compares the overall performance of reading and preprocessing images using OpenCV (CV2), Pillow (PIL), and Torchvision libraries.

- While Torchvision might offer the best performance when used independently, integrating it with a PyTorch-based data loader class might lead to better performance with OpenCV. 
- This code utilizes the hyperfine benchmarking tool (https://github.com/sharkdp/hyperfine) to evaluate each library's performance through individual runs.
- My computer configuration for testing is a Mac M1 Max with 64 GB of unified memory, 10 CPU cores, and macOS 10.15
