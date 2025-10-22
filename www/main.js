const SPEED_SLOW = 1;

const imagePaths0 = [];
let headEls;

for (let i = 0; i < 244; i++) {
  imagePaths0.push(`./imgs/2025-10-24/ml20251019_${("00000".concat(i)).slice(-5)}.jpg`);
}

const hour = (new Date()).getHours();
const offsetIdx = Math.floor((hour / 24) * imagePaths0.length);

const firstN = imagePaths0.slice(0, offsetIdx);
const imagePaths = imagePaths0.slice(offsetIdx).concat(firstN);

function slide() {
  const container = document.getElementById("main-container");
  const cML = parseInt(window.getComputedStyle(container)["marginLeft"]);
  const headElRect = headEls[0].getBoundingClientRect();

  if (headElRect.right < 0) {
    container.style.marginLeft = `${cML + headElRect.width}px`;
    container.appendChild(headEls[0]);
    headEls[4].querySelector(".img-hor").setAttribute("src", headEls[4].dataset.src);
    headEls = getHead();
  } else {
    container.style.marginLeft = `${cML - SPEED_SLOW}px`;
  }

  setTimeout(() => slide(), 50);
}

function getHead() {
  const container = document.getElementById("main-container");
  return Array.from(container.getElementsByClassName("img-container")).slice(0, 5);
}

window.addEventListener("load", () => {
  const container = document.getElementById("main-container");

  imagePaths.forEach((imgPath, i) => {
    const imgDivEl = document.createElement("div");
    imgDivEl.classList.add("img-container");
    imgDivEl.setAttribute("data-src", `${imgPath}`);
    container.appendChild(imgDivEl);

    const imgEl = document.createElement("img");
    imgEl.classList.add("img-hor");

    imgDivEl.appendChild(imgEl);
  });

  headEls = getHead();
  headEls.forEach(el => el.querySelector(".img-hor").setAttribute("src", el.dataset.src));

  setTimeout(() => slide(), 1000);
});
