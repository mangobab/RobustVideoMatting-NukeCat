# RVM for Nuke

## Introduction
This project implements **Robust Video Matting (RVM)** inside The Foundry's Nuke, enabling high-quality real-time background matting using deep learning.

This adaptation allows RVM to run natively within Nuke as a Cattery model, making it easy to integrate into existing compositing workflows without requiring additional dependencies.

---

## Features
- **High-quality real-time video matting** with detail preservation
- **Optimized deep learning model** using `rvm_mobilenetv3.pth`
- **Support for RGB and Alpha channels**, enabling direct compositing
- **Works on a wide range of backgrounds** without requiring trimaps
- **Lightweight and efficient**, allowing fast inference directly in Nuke

---

## Compatibility
- Tested on Nuke 14.1v5 and Linux

---

## Installation
1. **Download** and extract the latest release.
2. Copy the extracted `Cattery` folder to your `.nuke` directory or Nuke's plugin path.
3. In Nuke, go to **Cattery > Update**, or restart Nuke.
4. The RVM node will be accessible under **Cattery > Matting > RVM**.

---

## Model Information
This implementation uses the **MobileNetV3-based RVM model (`rvm_mobilenetv3.pth`)**, which is optimized for real-time performance while maintaining matting results.

For more details on the original model, visit: [Robust Video Matting GitHub](https://github.com/PeterL1n/RobustVideoMatting)

---

## Release Information
- **Version:** 1.0.1
- **Release Date:** *Feb.07.2025*
- **Developer:** Dean Kwon ([deankwon724@gmail.com](mailto:deankwon724@gmail.com))

---

## License & Acknowledgments
This project is adapted from [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) and follows the original licensing terms.

Users are responsible for ensuring compliance with licensing requirements when using the model and its dependencies.

