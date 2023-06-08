// App JS 
// Select the search button element
const searchButton = document.querySelector(".button1");

// Select the card container element
const cardContainer = document.querySelector(".card-container");

// Select the card element
const card = cardContainer.querySelector(".card");

const hotelOutput = document.querySelector(".card-title");
const hotelRating = document.querySelector(".card-rating ");
const hotelNumber = document.querySelector(".card-number");

// Get the preferences
const Preference1 = document.getElementById("preference1");
const Preference2 = document.getElementById("preference2");
const Preference3 = document.getElementById("preference3");

// Get API
const getPredictions = "http://127.0.0.1:3000/predict?";

let outHotels;

async function getHotels(pref1, pref2, pref3) {
  const out = fetch(
    getPredictions + `pref1=${pref1}&pref2=${pref2}&pref3=${pref3}`
  )
    .then((res) => res.json())
    .then((res) => res);
  return out;
}

// console.log(outHotels);

// Add an event listener to the search button
searchButton.addEventListener("click", async () => {
  console.log(Preference1.value, Preference2.value, Preference3.value);

  if (Preference1.value === "Select") {
    alert("Select atleast one preference");
  }

  cardContainer.innerHTML = "";
  output = await getHotels(
    Preference1.value,
    Preference2.value,
    Preference3.value
  );
  const totalRecommendations = 5;
  for (let i = 0; i < totalRecommendations; i++) {
    cardContainer.style.display = "flex";
    hotelOutput.innerText = output[i][0];
    hotelRating.innerText = "Rating:" + " " + [output[i][2]];
    hotelNumber.innerText = "Preference" + " " + (i + 1);
    const clonedCard = card.cloneNode(true);
    clonedCard.style.display = "inline-block";
    cardContainer.appendChild(clonedCard);
  }
});
