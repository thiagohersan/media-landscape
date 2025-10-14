const SPEED_SLOW = 1;
const SPEED_NORM = 2;
const SPEED_FAST = 4;

const imagePaths = [];
const imageSpeeds = [];
const imageEls = [];

let nImgIdx = 1;
let cSpeed = SPEED_FAST;
let tSpeed = SPEED_FAST;

for (let i = 1; i < 9; i++) {
  imagePaths.push(`./imgs/text_${("00".concat(i)).slice(-2)}.jpg`);
  imageSpeeds.push(SPEED_FAST);
}

imagePaths.push("./imgs/landscape_00.jpg")
imageSpeeds.push(SPEED_FAST);
for (let i = 1; i < 4; i++) {
  imagePaths.push(`./imgs/apocalypse_${("00".concat(i)).slice(-2)}.jpg`);
  imageSpeeds.push(SPEED_FAST);
}

function elementPresent(element) {
  const rect = element.getBoundingClientRect();
  return (
    rect.left <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

function slide() {
  if (nImgIdx < imageEls.length) {
    if (elementPresent(imageEls[nImgIdx])) {
      console.log("new element", nImgIdx);
      tSpeed = imageEls[nImgIdx].getAttribute("data-speed");
      nImgIdx = (nImgIdx + 1);
    }

    cSpeed = 0.995 * cSpeed + 0.005 * tSpeed;

    const container = document.getElementById('main-container');
    const cML = parseInt(window.getComputedStyle(container)["marginLeft"]);
    container.style.marginLeft = `${cML - parseInt(cSpeed)}px`;
    requestAnimationFrame(slide);
  } else {
    tSpeed = 0;
  }
}

window.addEventListener('load', () => {
  const container = document.getElementById('main-container');
  const startButt = document.getElementById('start-button');

  imagePaths.forEach((imgPath, i) => {
    const imgDivEl = document.createElement("div");
    imgDivEl.classList.add("img-container");
    imgDivEl.setAttribute("data-speed", imageSpeeds[i]);
    container.appendChild(imgDivEl);

    const imgEl = document.createElement("img");
    imgEl.setAttribute("src", `${imgPath}`);
    imgEl.classList.add("img-hor");
    imgDivEl.appendChild(imgEl);

    imageEls.push(imgDivEl);
  });

  startButt.addEventListener("click", (ev) => {
    ev.target.style.display = "none";
    requestAnimationFrame(slide);
  });
});
