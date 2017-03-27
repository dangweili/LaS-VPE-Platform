/*
 * This file is part of LaS-VPE Platform.
 *
 * LaS-VPE Platform is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * LaS-VPE Platform is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with LaS-VPE Platform.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.cripac.isee.vpe.data;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.log4j.Level;
import org.cripac.isee.alg.pedestrian.attr.Attributes;
import org.cripac.isee.alg.pedestrian.tracking.Tracklet;
import org.cripac.isee.vpe.common.DataType;
import org.cripac.isee.vpe.ctrl.SystemPropertyCenter;
import org.cripac.isee.vpe.ctrl.TaskData;
import org.cripac.isee.vpe.debug.FakeRecognizer;
import org.cripac.isee.vpe.debug.FakePedestrianTracker;
import org.cripac.isee.vpe.util.logging.ConsoleLogger;
import org.junit.Before;

import java.util.Properties;
import java.util.UUID;

import static org.cripac.isee.vpe.util.SerializationHelper.serialize;
import static org.cripac.isee.vpe.util.kafka.KafkaHelper.sendWithLog;

/**
 * This is a JUnit test for the DataManagingApp.
 * Different from usual JUnit tests, this test does not initiate a DataManagingApp.
 * The application should be run on YARN in advance.
 * This test only sends fake data messages to and receives results
 * from the already running application through Kafka.
 * <p>
 * Created by ken.yu on 16-10-31.
 */
public class DataManagingAppTest {

    private KafkaProducer<String, byte[]> producer;
    private ConsoleLogger logger;

    public static void main(String[] args) {
        DataManagingAppTest test = new DataManagingAppTest();
        try {
            test.init(args);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
        try {
            test.testTrackletSaving();
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            test.testAttrSaving();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Before
    public void init() throws Exception {
        init(new String[]{"-a", DataManagingApp.APP_NAME,
                "--system-property-file", "conf/system.properties",
                "--app-property-file", "conf/" + DataManagingApp.APP_NAME + "/app.properties",
                "-v"});
    }

    public void init(String[] args) throws Exception {
        SystemPropertyCenter propCenter = new SystemPropertyCenter(args);

        Properties producerProp = propCenter.getKafkaProducerProp(false);
        producer = new KafkaProducer<>(producerProp);
        logger = new ConsoleLogger(Level.DEBUG);
    }

    //    @Test
    public void testTrackletSaving() throws Exception {
        TaskData.ExecutionPlan plan = new TaskData.ExecutionPlan();
        TaskData.ExecutionPlan.Node savingNode = plan.addNode(DataManagingApp.IDRankSavingStream.OUTPUT_TYPE);

        Tracklet[] tracklets = new FakePedestrianTracker().track(null);
        String taskID = UUID.randomUUID().toString();
        for (int i = 0; i < tracklets.length; ++i) {
            Tracklet tracklet = tracklets[i];
            tracklet.id = new Tracklet.Identifier("fake", i);

            TaskData data = new TaskData(
                    savingNode.createInputPort(DataManagingApp.TrackletSavingStream.PED_TRACKLET_SAVING_PORT),
                    plan,
                    tracklet);
            sendWithLog(DataType.TRACKLET.name(),
                    taskID,
                    serialize(data),
                    producer,
                    logger);
        }
    }

    //    @Test
    public void testAttrSaving() throws Exception {
        TaskData.ExecutionPlan plan = new TaskData.ExecutionPlan();
        TaskData.ExecutionPlan.Node savingNode = plan.addNode(DataManagingApp.IDRankSavingStream.OUTPUT_TYPE);

        Attributes attributes = new FakeRecognizer().recognize(
                new FakePedestrianTracker().track(null)[0]);
        attributes.trackletID = new Tracklet.Identifier("fake", 0);

        TaskData data = new TaskData(
                savingNode.createInputPort(DataManagingApp.AttrSavingStream.PED_ATTR_SAVING_PORT),
                plan,
                attributes);
        sendWithLog(DataType.ATTRIBUTES.name(),
                UUID.randomUUID().toString(),
                serialize(data),
                producer,
                logger);
    }
}
