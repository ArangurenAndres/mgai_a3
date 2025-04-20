## MGAI project


## Setup

You can run the following bash lines to create a python env, activate it and install its dependencies

1. Create python env

```sh
chmod +x setup.sh
./setup.sh
```



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

