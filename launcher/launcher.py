import os
import argparse

# usage example: python launcher.py --cuda_file=mem.cu --uuid=MIG-14cb47da-9f77-5625-9226-ea5afac27f09
def launch(cuda_file, mig_uuid):
    cuda_program = cuda_file[:-3]
    os.popen('sh ./launcher.sh {} {} {} {}'.format(cuda_file, cuda_program, mig_uuid, cuda_program))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cuda_launcher')
    parser.add_argument('-c', '--cuda_file', help='cuda file name', required=True)
    parser.add_argument('-u', '--uuid', help='MIG UUID', required=True)

    args = vars(parser.parse_args())

    launch(args['cuda_file'], args['uuid'])
