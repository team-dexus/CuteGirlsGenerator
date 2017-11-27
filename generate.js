
function generate(){ //Added by mirukuma ,Changed by JPNYKW ,Changed by mirukuma ,Changed by JPNYKW
  var pic=d.getElementById('GirlPic');
  var item=['MG7.png','MG57.png','code100.png']
  var text=d.getElementById('words').value;
  var arrayRes=d.getElementById('ArrayResult');
  var result='===WORDS===<br>';
  tag(text).forEach(function(e,i,a){result+=`<span id=res${i}>${e}</span><br>`});
  d.getElementById('ShowText').textContent='Generation result in '+text;
  d.getElementById('ArrayRes').innerHTML=`${result}=========`;
  pic.src=item[~~(Math.random()*3)];


}

/*
function generate2() { //Added by mirukuma
  const model = new KerasJS.Model({ //Added by mirukuma
    filepaths: {
      model: './data/generator.json',
      weights: './data/generator_weights.buf',
      metadata: './data/generator_metadata.json'
    },
    gpu: true
  })
  console.log(model)

  model.ready()
    .then(() => {
      // input data object keyed by names of the input layers
      // or `input` for Sequential models
      // values are the flattened Float32Array data
      // (input tensor shapes are specified in the model config)
      var noise= new Float32Array(100)
      for(i=0; i<100; i=i+1){
        noise[i]=Math.random()*2-1
      }
      const inputData = {
        'input': noise
      }

      console.log(inputData)
      // make predictions
      return model.predict(inputData)
    })
    .then(outputData => {
      // outputData is an object keyed by names of the output layers
      // or `output` for Sequential models
      // e.g.,
      // outputData['fc1000']
      generatedData=outputData.output; // Added by JPNYKW
      console.log(generatedData);  // Added by JPNYKW
      drawGeneratedImage();  // Added by JPNYKW
    })
    .catch(err => {
      console.log(err)
    })

}
*/
// ES6(modern JavaScript) version
function generate2(){
  onGenerated=false; //Added by JPNYKW
  console.log(onGenerated); //Added by JPNYKW
  console.log('generate2');
  WebDNN.load('./output')
     .then(function(runner){
         console.log('loaded');
         console.log(runner.backendName);
         let x = runner.getInputViews()[0];
         let x2 = runner.getInputViews()[1];
         console.log(x)
         let y = runner.getOutputViews()[0];

         var noise= new Float32Array(100)

         for(i=0; i<100; i=i+1){
           noise[i]=Math.random()*2-1
         }

         numberOfTag=1539

         var tags= new Float32Array(numberOfTag)
         /*
         for(i=0; i<1539; i=i+1){
           tags[i]=Math.random()*2-1
         }
         タグの指定
         */
         x.set(noise);
         x2.set(tags)
         console.log(x)
         runner.run()
            .then(function() {
              console.log('finished');
              let y_typed_array = y.toActual();
              console.log(y_typed_array);
              generatedData=y_typed_array; //Added by JPNYKW
              drawGeneratedImage(); //Added by JPNYKW
              onGenerated=true; //Added by JPNYKW
              d.getElementById("download").href=d.getElementById("output").toDataURL()  //Added by CS017
            });

         // add your code here.
     });


}
