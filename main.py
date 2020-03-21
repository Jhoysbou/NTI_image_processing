from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl


kernel = '/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel'
session = WolframLanguageSession(kernel=kernel)

sample = session.evaluate(wl.RandomVariate(wl.NormalDistribution(0,1), 1e6))
print(sample)