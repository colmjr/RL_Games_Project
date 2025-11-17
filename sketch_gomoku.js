const SIZE = 15;
const CELL = 30;
const WIDTH = SIZE * CELL;
const HEIGHT = SIZE * CELL;
const API = "http://localhost:8000";
let board = Array(SIZE * SIZE).fill(0);
let gameOver = false;
let pending = false;

async function fetchState() {
  const res = await fetch(`${API}/state`);
  const data = await res.json();
  board = data.board;
  gameOver = data.done;
  redrawBoard();
}

async function sendMove(row, col) {
  if (gameOver || pending) return;
  pending = true;
  const res = await fetch(`${API}/move`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({row, col})
  });
  const data = await res.json();
  board = data.board;
  gameOver = data.done;
  redrawBoard();
  pending = false;
}

function setup() {
  createCanvas(WIDTH, HEIGHT);
  rectMode(CENTER);
  colorMode(RGB, 255);
  fetchState();
}

function draw() {
  redrawBoard();
}

function draw() {
  background(215, 185, 96);
  fill(0);
  for (let i = 0; i <= WIDTH; i += CELL) {
    line(i, 0, i, HEIGHT);
  }
  for (let j = 0; j <= HEIGHT; j += CELL) {
    line(0, j, WIDTH, j);
  }
  for (let idx = 0; idx < board.length; idx += 1) {
    const val = board[idx];
    if (val === 0) continue;
    const x = idx % SIZE;
    const y = Math.floor(idx / SIZE);
    drop(x, y, Math.round(val * 2));
  }
}
function drop(x, y, symbol) {//symbol is 1 for black, 2 white
    fill((2-symbol)*255);
    circle(x,y,25)
}

function mousePressed() {
  const [x, y] = human_encoder(mouseX, mouseY);
  sendMove(y, x);
}

function drop(x, y, symbol) {
  fill((2 - symbol) * 255);
  circle(x * CELL + CELL / 2, y * CELL + CELL / 2, CELL - 5);
}

function human_encoder(px, py) {
  return [Math.floor(px / CELL), Math.floor(py / CELL)];
}
