const _GRIDWIDTH=15;
const _GRIDHEIGHT=15;
const GRIDWIDTH=_GRIDWIDTH*30
const GRIDHEIGHT=_GRIDHEIGHT*30
function setup() {
  createCanvas(GRIDWIDTH, GRIDHEIGHT);
  rectMode(CENTER);
  colorMode(RGB,255);
  background(215, 185, 96);
}

function draw() {
  fill(0);
  for (let i = 0; i<=GRIDWIDTH; i+=30){
    line(i,0,i,GRIDHEIGHT);
    
  }
  for (let j = 0; j<=GRIDHEIGHT; j+=30){
    line(0,j,GRIDWIDTH,j);
  }
  drop(8,2,1)
  drop(3,7,2)
  drop(8,3,1)
  drop(4,6,2)
}
function drop(x,y,symbol){//gomoku symbol mapping:1=white,2=black
  fill((2-symbol)*255);
  circle(x*30,y*30,25);
}