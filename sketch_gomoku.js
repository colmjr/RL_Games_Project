const SIZE = 15;
const CELL = 30;
const API = "http://localhost:8000";
let board = Array(SIZE * SIZE).fill(0);
let gameOver = false;
async function fetchState() {
  const res = await fetch(`${API}/state`);
  const data = await res.json();
  board = data.board;
  gameOver = data.done;
  redrawBoard();
}

async function sendMove(row, col) {
  if (gameOver) return;
  const res = await fetch(`${API}/move`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({row, col})
  });
  const data = await res.json();
  board = data.board;
  gameOver = data.done;
  redrawBoard();
}
function setup(){
  createCanvas(GRIDWIDTH, GRIDHEIGHT);
  rectMode(CENTER);
  colorMode(RGB,255);
  background(215, 185, 96);
}
function redrawBoard() {
  fill(0);
  for (let i = 0; i<=GRIDWIDTH; i+=30){
    line(i,0,i,GRIDHEIGHT);
  }
  for (let j = 0; j<=GRIDHEIGHT; j+=30){
    line(0,j,GRIDWIDTH,j);
  }
  // draw grid…
  for (let idx = 0; idx < board.length; idx++) {
    const val = board[idx];           // 0 empty, 0.5 agent O, 1 human X
    if (val === 0) continue;
    const x = idx % SIZE;
    const y = Math.floor(idx / SIZE);
    drop(x, y, Math.round(val * 2));  // convert back to {0,1,2}
  }
}

function mousePressed() {
  const [x, y] = human_encoder(mouseX, mouseY);
  sendMove(y, x);                     // convert to row/col if needed
}
