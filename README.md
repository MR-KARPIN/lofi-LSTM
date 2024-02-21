# Bachelor's degree in Computer Engineering Final Project

This proyect is the final project of a bachelor's in Computer engineering graded a 99/100. It is an implementation of an Artificial Inteligence generator of notes of Lo-Fi music. It applys LSTM (Long Short-Term Memory) layers to provide temporal context.

Given a number x of seconds (duration of the song) and a number y of notes that should be played in the song (Each note has an offset of 1.5 between them by default so y should be greater than 1,2*x), it generates you a Lo-Fi song which contains a set of generated notes, a beat and ambient rain.

## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

have installed the following packages:
  1. tensorflow
  2. music21
  3. dawdreamer
  4. pydub

### Installing

1. Download the repository
2. Run generator.py

## Contributing

Please write me to pxpshack@gmail.com if you want to contact me for the improvement of the project or more insight.

## Authors

  - **Antonio Carpintero Castilla**

## License

This project is license of the Universidad Politécnica de Madrid

## Acknowledgments

  - Thanks to [Zachary](https://ai.plainenglish.io/building-a-lo-fi-hip-hop-generator-e24a005d0144) for the dataset and their project
