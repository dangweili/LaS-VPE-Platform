/**
 * To test the function getTrackletInfo in HadoopHelper
 *
 * @Author da.li on 2017-04-17
 */

package org.cripac.isee.vpe.util;

import java.io.IOException;
import java.net.URISyntaxException;

import static org.cripac.isee.vpe.util.hdfs.HadoopHelper.getTrackletInfo;


public class GetTrackletInfoTest {

    public static void main(String[] args) throws Exception {
     
        // It is necessary to change the host:port and directory as your practical situation.
//        String base = "har://hdfs-kman-nod2:8020";
        String storeDir = "/user/vpe.cripac/metadata_ssd_tracker_0609/20131223102739-20131223103331/27369a4b-8588-44b8-ab98-629bdf96a9d6.har";
//        storeDir = base + storeDir;

        try {
            String trackletInfo = getTrackletInfo(storeDir);
            System.out.println("Test getTrackletInfo, the info is:");
            System.out.println(trackletInfo);
        } catch(IOException e1) {
        } catch(URISyntaxException e2) {
        }
    }
}
