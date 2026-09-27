"""Remove only recorded CAROM actors; never reload a user's existing world."""
import argparse
import json
from pathlib import Path
import carla

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    record=json.loads((a.run/'run.json').read_text(encoding='utf-8'))
    client=carla.Client('127.0.0.1',2000);client.set_timeout(15);world=client.get_world()
    for actor_id in record['kept_actor_ids']:
        actor=world.get_actor(actor_id)
        if actor and actor.attributes.get('role_name')=='carom_baseline':
            print(actor_id,actor.destroy())
