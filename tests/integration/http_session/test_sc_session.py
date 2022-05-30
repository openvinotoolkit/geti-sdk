from sc_api_tools.http_session import SCSession


class TestSCSession:
    def test_authenticate(self, fxt_sc_session: SCSession):
        """
        Test that the authenticated SCSession instance contains authentication cookies
        """
        fxt_sc_session.authenticate(verbose=False)

    def test_product_version(self, fxt_sc_session: SCSession):
        """
        Test that the 'version' attribute of the session is assigned a valid product
        version
        """
        possible_versions = ['1.0', '1.1', '1.2']
        version_matches = [
            fxt_sc_session.version.startswith(version) for version in possible_versions
        ]
        assert sum(version_matches) == 1
