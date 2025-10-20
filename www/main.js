const SPEED_SLOW = 1;

const imagePaths = [];
let headEls;

for (let i = 0; i < 112; i++) {
  imagePaths.push(`./imgs/ml20251019_${("00000".concat(i)).slice(-5)}.jpg`);
}

// TODO: use modulus and shift the paths by current hour
// 12am = 0%, 6am = 25%, 12pm = 50%, 6pm = 75%

function slide() {
  const container = document.getElementById("main-container");
  const cML = parseInt(window.getComputedStyle(container)["marginLeft"]);
  const headElRect = headEls[0].getBoundingClientRect();

  if (headElRect.right < 0) {
    container.style.marginLeft = `${cML + headElRect.width}px`;
    container.appendChild(headEls[0]);
    headEls[2].querySelector(".img-hor").setAttribute("src", headEls[2].dataset.src);
    headEls = getHead();
  } else {
    container.style.marginLeft = `${cML - SPEED_SLOW}px`;
  }

  setTimeout(() => slide(), 50);
}

function getHead() {
  const container = document.getElementById("main-container");
  return Array.from(container.getElementsByClassName("img-container")).slice(0, 3);
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
