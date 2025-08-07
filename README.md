
# 🪰 Split-GAL4 Matcher

**Split-GAL4 Matcher** is an interactive web app for visualizing and evaluating binary expression matches between Drosophila hemidriver lines (AD and DBD) using FlyLight Gen1 CDM images. Users can input any two lines and generate brain and/or VNC match images, along with expression grading and a confidence score.


## Features

- Search AD and DBD lines by name
- Generate match images (brain and/or VNC)
- View grayscale, cluster, and histogram visualizations
- Get a numeric match confidence score with an interpretive grade


## Try the App
[Split-GAL4 Matcher](https://split-gal4-matcher.streamlit.app/)


## Run Locally

Clone the project

```bash
  git clone https://github.com/cintiashamsu/split-gal4-matcher
```

Go to the project directory

```bash
  cd split-gal4-matcher
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the app

```bash
  streamlit run app.py
```

