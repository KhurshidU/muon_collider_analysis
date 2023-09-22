import os
import subprocess

# Shell script content

for i in range(8100,10100,100):
        runi_content = f"""
        #!/bin/bash
        Marlin --global.LCIOInputFiles=/work/khurriccan/MuC-Tutorial/simulation/mumu_nunuH_1point5tev_hsbib_all.slcio --Output_REC.LCIOOutputFile=Output_REC_bib_1point5tev_hsbib_{i}.slcio  --Output_DST.LCIOOutputFile=Output_DST_bib_1point5tev_hsbib_{i}.slcio  --MyAIDAProcessor.FileName=histograms_bib_1point5tev_hsbib_{i}  --global.MaxRecordNumber=100  --global.SkipNEvents={i-100} --Config.Overlay=Test steer_reco_bib_1point5tev_hsbib_100.xml > reco_steer_reco_bib_1point5tev_hsbib_{i}.log 2>&1 &
        """
        # Write the shell script file
        run_script_filename = f'run_local_{i}.sh'
        with open(run_script_filename, 'w') as script_file:
              script_file.write(runi_content)
        os.chmod(run_script_filename, 0o755)
        # Make the shell script file executable
        # Execute the shell script
        os.system(f'./{run_script_filename}')
