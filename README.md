This project is based on the excellent work of MiniGPT-3D. The original codebase and paper can be found at: https://github.com/TangYuan96/MiniGPT-3D.


I tried to adapt the original project with the RTX 5090 and tested various CUDA versions, yet compatibility issues persisted. Hence, I gave up the attempt.

It is found that the hard-coded online weight paths in the official code are inaccessible. You need to download the weights from this link: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth. Modify the corresponding part in the file /MiniGPT-3D/minigpt4/models/minigpt_v2.py to point it to the local weights. 

'''
self.load_from_pretrained(
            url_or_filename="…"
)
'''

The subjective evaluation API used in the original project is qwen2-72b-instruct, which has been offline and can no longer be called. I have modified the code to enable calls to Qwen-Flash now.

After completing the training in accordance with the process, conduct evaluation using the self-trained checkpoints, and the evaluation results are satisfactory.

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All modifications and additions in this repository are released under the same license.