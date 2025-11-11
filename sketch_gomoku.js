const _GRIDWIDTH=15;
const _GRIDHEIGHT=15;
const GRIDWIDTH=_GRIDWIDTH*30
const GRIDHEIGHT=_GRIDHEIGHT*30
function setup() {
  createCanvas(GRIDWIDTH, GRIDHEIGHT);
  rectMode(CENTER);
  colorMode(RGB,255);
  background(215, 185, 96);
  fill(0);
  for (let i = 0; i<=GRIDWIDTH; i+=30){
    line(i,0,i,GRIDHEIGHT);
    
  }
  for (let j = 0; j<=GRIDHEIGHT; j+=30){
    line(0,j,GRIDWIDTH,j);
  }
}

function draw() {
  if (mouseIsPressed){
    let [x,y]= human_encoder(mouseX,mouseY);
    drop(x,y,2); 
    //human is always first because PPO is deterministic and so humans introduce more variability in the beginnings of games
  }
}
function drop(x,y,symbol){//gomoku symbol mapping:1=white,2=black
  fill((2-symbol)*255);
  circle(x*30,y*30,25);
}
function human_encoder(x,y){
  return [Math.round(x/30),Math.round(y/30)];
}