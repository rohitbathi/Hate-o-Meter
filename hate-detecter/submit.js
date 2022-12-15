async function submitHandler() {
    const req_text = document.getElementById("inputText").value;
    // console.log(req_text);
    const predictionEle = document.getElementById("prediction");

    let data = { inp_text: req_text };

    const finalData = await fetchPred(data);
    predictionEle.innerHTML = finalData.prediction;


    if(finalData.score<0.45){
        predictionEle.classList.add('normal');
        predictionEle.classList.remove('hate')
    }else{
        predictionEle.classList.add('hate');
        predictionEle.classList.remove('normal');
    }
}

async function fetchPred(data) {
    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        mode: "cors",
        cache: "no-cache",
        headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": true,
        },
        body: JSON.stringify(data),
    });
    const resData = await response.json();
    return resData;
}
