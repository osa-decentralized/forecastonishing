[![Build Status](https://travis-ci.org/osahp/forecastonishing.svg?branch=master)](https://travis-ci.org/osahp/forecastonishing)
[![codecov](https://codecov.io/gh/osahp/forecastonishing/branch/master/graph/badge.svg)](https://codecov.io/gh/osahp/forecastonishing)
[![Maintainability](https://api.codeclimate.com/v1/badges/62ba0c41d25448bdbaac/maintainability)](https://codeclimate.com/github/osahp/forecastonishing/maintainability)

# forecastonishing

## What is it?
This repo contains easy-to-use tools for forecasting. Currently, the list of provided utilities consists of:
* On-the-fly selector, an adaptive selector that can leverage abilities of robust, yet extremely simple methods such as moving average. More details can be found in [a tutorial](https://github.com/osahp/forecastonishing/blob/master/docs/on_the_fly_selector_demo.ipynb);
* Some auxiliary classes and functions that can make forecasting easier.

## How to install the package?
To install the package in a virtual environment named `your_virtual_env`, run this from your terminal:
```
cd path/to/your/destination
git clone https://github.com/osahp/forecastonishing
cd forecastonishing
source activate your_virtual_env
pip install .
```
