# VisLoc: Visual Localization Package

VisLoc is a Python package that implements state-of-the-art visual localization pipelines using SuperGlue and OmniGlue mappers. The package provides tools to match drone images against a map database and estimate the drone's location. It was prepared as a graduation project. - ITU UUBF -

## Features

- SuperGlue pipeline implementation for feature matching and localization
- OmniGlue pipeline implementation for feature matching and localization
- Support for drone image processing and map database matching
- Visualization tools for match results and localization accuracy
- Comprehensive logging and metrics tracking
- Configurable parameters for both pipelines

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and setup model weights:
   - Download the model weights from [here](https://drive.google.com/file/d/1znZQYihtus5DGZDhwl_TY1a57Mra3kCP/view?usp=sharing)
   - Extract the contents to `visloc/models` directory
   - Download SuperGlue weights from [here](https://drive.google.com/file/d/1g1pjdRa4ekCHNFBPq4s-wP-d2lMHt1U1/view?usp=drive_link)
   - Extract the SuperGlue weights to `src/superglue/weights` directory

3. Run the localization pipeline:
```bash
python main.py --drone_image path/to/drone/image.jpg --map_dir path/to/map/images
```


## TO DO List
- [ ] Supporting LoFTR, MINIMA, LightGlue and the SOTA matchers
- [ ] Add support for different coordinate systems
- [ ] Implement additional visualization tools
- [ ] Add support for different map database formats
- [ ] Add example notebooks and tutorials
- [ ] ROS Implementation
- [ ] C++ Implementation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- https://github.com/magicleap/SuperGluePretrainedNetwork
- https://github.com/google-research/omniglue
- Special thanks to all contributors and users of this package 