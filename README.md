# MGAI project


## Create requirements.txt file 


pip install pipreqs

# From the root of your project directory:



```sh
pip install pipreqs

pipreqs . --force



```



## Setup

You can set up the Python environment and install dependencies using the following commands:

1. For Unix/Linux/MacOS:
```sh
chmod +x setup.sh
./setup.sh
```

2. For Windows:
Either double-click the `setup.bat` file or run it from the command prompt:
```cmd
setup.bat
```

Alternatively, you can run these commands manually:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. If you are a collaborator in this project before doing any modifications create a new branch and checkout :) merci 


## Project

The following project is divided in two main taks:

1. Implement GAN and Diffusion models to create realistic Super Mario levels 

2. Train agent to autonomously play the game without previous knowledge


## Level generation

The data used as input for the GAN and Diffusion model will plain txt which is then rendered to (RGB) channels using the legend.json file. We use txt since we are not performing any image analysis task such as segmentation or classification in which color must be kept. 

```sh
{
    "tiles" : {
        "X" : ["solid","ground"],
        "S" : ["solid","breakable"],
        "-" : ["passable","empty"],
        "?" : ["solid","question block", "full question block"],
        "Q" : ["solid","question block", "empty question block"],
        "E" : ["enemy","damaging","hazard","moving"],
        "<" : ["solid","top-left pipe","pipe"],
        ">" : ["solid","top-right pipe","pipe"],
        "[" : ["solid","left pipe","pipe"],
        "]" : ["solid","right pipe","pipe"],
        "o" : ["coin","collectable","passable"],
        "B" : ["Cannon top","cannon","solid","hazard"],
        "b" : ["Cannon bottom","cannon","solid"]
    }
}
```




## Agent for autonomous playing both real and generated levels