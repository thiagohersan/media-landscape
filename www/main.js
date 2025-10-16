const SPEED_SLOW = 1;

const imagePaths = [];
const imageEls = [];

let nImgIdx = 1;

for (let i = 0; i < 8; i++) {
  imagePaths.push(`./imgs/test00_${("00".concat(i)).slice(-2)}.jpg`);
}
for (let i = 0; i < 9; i++) {
  imagePaths.push(`./imgs/test01_${("00".concat(i)).slice(-2)}.jpg`);
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
      nImgIdx = (nImgIdx + 1);
    }

    const container = document.getElementById('main-container');
    const cML = parseInt(window.getComputedStyle(container)["marginLeft"]);
    container.style.marginLeft = `${cML - SPEED_SLOW}px`;
    setTimeout(() => slide(), 100);
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
    container.appendChild(imgDivEl);

    const imgEl = document.createElement("img");
    imgEl.setAttribute("src", `${imgPath}`);
    imgEl.classList.add("img-hor");

    if (i == 0) {
      imgEl.style.width = `${window.innerWidth / 1.1}px`;
    }

    imgDivEl.appendChild(imgEl);

    imageEls.push(imgDivEl);
  });

  startButt.style.display = "none";
  startButt.addEventListener("click", (ev) => {
    ev.target.style.display = "none";
    requestAnimationFrame(slide);
  });

  const urlHash = window.location.hash.toLowerCase();
  console.log(urlHash);
  if (!urlHash.includes("stop")) {
    setTimeout(() => requestAnimationFrame(slide), 500);
  }
});
