Evaluating training set
src=[Det är gammal skåpmat .], target=[That 's old news .], predicted=[That 's old cold .]
src=[Jag såg ett idag .], target=[I saw one today .], predicted=[I saw a today .]
src=[Jag kan fortfarande kämpa .], target=[I can still fight .], predicted=[I can still fight .]
src=[Spara det till senare .], target=[Save it till later .], predicted=[Save it till later .]
src=[Jag ska hämta min kappa .], target=[I 'll get my coat .], predicted=[I 'll get my coat .]
src=[Jag blev Toms vän .], target=[I became Tom 's friend .], predicted=[I saw Tom 's time .]
src=[Har du förrått oss ?], target=[Have you betrayed us ?], predicted=[Have you betrayed us ?]
src=[Jag har massor med vänner .], target=[I have lots of friends .], predicted=[I have lots of ideas .]
src=[Jag litade på Tom .], target=[I relied on Tom .], predicted=[I relied Tom Tom .]
src=[Jag kan inte göra det i dag .], target=[I ca n't do it today .], predicted=[I ca n't do this this .]
BLEU-1: 0.276801
BLEU-2: 0.526119
BLEU-3: 0.680222
BLEU-4: 0.725341
Evaluating testing set
src=[Är du på biblioteket ?], target=[Are you in the library ?], predicted=[Are you on yet ?]
src=[Gör dig redo för avfärd .], target=[Get ready to leave .], predicted=[Get and to relax .]
src=[Vad finns i lådan ?], target=[What 's in the box ?], predicted=[What 's in problem ?]
src=[Behöver du hjälp ?], target=[Do you need a hand ?], predicted=[Do you need anything ?]
src=[Jag måste hålla med .], target=[I have to agree .], predicted=[I have to to go .]
src=[Tom låter utmattad .], target=[Tom sounds exhausted .], predicted=[Tom looked puzzled .]
src=[En tupplur vore bra .], target=[A nap would be good .], predicted=[The one was too . .]
src=[Vi hoppas att det här stämmer .], target=[We hope this is true .], predicted=[We hope this this true .]
src=[Vad skulle ha veta ?], target=[What would he know ?], predicted=[What could they ?]
src=[De bor i närheten .], target=[They live nearby .], predicted=[They 'll better .]
BLEU-1: 0.279074
BLEU-2: 0.528275
BLEU-3: 0.681893
BLEU-4: 0.726825


= Training

With 16000 lines:

English Vocabulary Size: 4672
English Max Length: 15
Other Vocabulary Size: 6831
Other Max Length: 15
Total params: 4,000,064
Trainable params: 4,000,064

Initial epochs took 159s dropped to 150s by epoch 19/30
Model val_loss = 1.32258

BLEU-1: 0.236563
BLEU-2: 0.486377
BLEU-3: 0.648909
BLEU-4: 0.697408
