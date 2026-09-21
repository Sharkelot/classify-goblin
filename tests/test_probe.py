"""Capability probe and versioned modality manifest (V1/V2)."""
import unittest

from jev_laya_free.multimodal import probe as probe_mod


class ProbeTests(unittest.TestCase):
    def test_manifest_structure_and_states(self):
        manifest = probe_mod.default_manifest()
        self.assertEqual(manifest['version'], 1)
        self.assertIn('capabilities', manifest)
        self.assertIn('probed_at', manifest)
        for name in ('text', 'image', 'video', 'audio', 'pdf'):
            self.assertIn(name, manifest['capabilities'])
            state = manifest['capabilities'][name]
            self.assertIn(state['state'], ('verified', 'unavailable', 'not_verified'))

    def test_local_observations_from_probe_report(self):
        # JEV-MM-01 local observations (not upstream claims).
        manifest = probe_mod.default_manifest()
        caps = manifest['capabilities']
        self.assertEqual(caps['text']['state'], 'verified')
        self.assertEqual(caps['image']['state'], 'verified')
        self.assertEqual(caps['video']['state'], 'verified')
        self.assertEqual(caps['audio']['state'], 'unavailable')
        self.assertEqual(caps['pdf']['state'], 'unavailable')

    def test_probe_report_hash_recorded(self):
        manifest = probe_mod.default_manifest()
        self.assertIn('evidence', manifest)
        self.assertTrue(manifest['evidence']['probe_report_sha256'])

    def test_upstream_claims_never_local_support(self):
        # A modality with no local probe must be not_verified, never verified.
        manifest = probe_mod.default_manifest()
        for name, state in manifest['capabilities'].items():
            if state['state'] == 'verified':
                self.assertIn(name, ('text', 'image', 'video'))

    def test_is_available(self):
        manifest = probe_mod.default_manifest()
        self.assertTrue(probe_mod.is_available(manifest, 'image'))
        self.assertFalse(probe_mod.is_available(manifest, 'audio'))
        self.assertFalse(probe_mod.is_available(manifest, 'pdf'))


if __name__ == '__main__':
    unittest.main()
